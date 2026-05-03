import { Parser } from '../vendor/pickleparser/src/index';
import { Tensor } from '../tensor/Tensor';
import { tensor } from '../ops/creation';
import { readBuffer } from '../backend';
import { DType, getDTypeBytes, getTypedArrayConstructor } from '../dtype';
import * as zip from '../vendor/zip.js/index.js';

const MODEL_FILENAME = 'model.json'; // Our custom JSON-based format

interface TensorMetadata {
  __tensor__: true;
  id: string;
  shape: number[];
  dtype: DType;
  requires_grad: boolean;
  device: string;
}

/**
 * Serialize a state dictionary (or any object containing tensors) to a ZIP Blob.
 * This currently uses a custom JSON-based format within the zip.
 *
 * TODO: For PyTorch compatibility, this should ideally create a `data.pkl` file
 *       using a Python-compatible pickler, replacing Tensors with references to
 *       binary data files (e.g., `archive/data/X`). This requires a TypeScript
 *       pickler implementation.
 */
export async function serializeToZip(obj: any): Promise<Blob> {
  const tensorMap = new Map<string, Tensor>();
  let tensorCount = 0;

  // 1. Traverse and replace tensors with metadata
  function replacer(key: string, value: any): any {
    if (value instanceof Tensor) {
      const id = String(tensorCount++);
      tensorMap.set(id, value);
      const meta: TensorMetadata = {
        __tensor__: true,
        id,
        shape: Array.from(value.shape),
        dtype: value.dtype,
        requires_grad: value.requires_grad,
        device: value.device,
      };
      return meta;
    }
    return value;
  }

  const jsonStr = JSON.stringify(obj, replacer, 2);

  // 2. Create Zip
  const blobWriter = new zip.BlobWriter("application/zip");
  const writer = new zip.ZipWriter(blobWriter);

  // 3. Write model.json
  await writer.add(MODEL_FILENAME, new zip.TextReader(jsonStr));

  // 4. Write tensors
  for (const [id, t] of tensorMap) {
    const data = await readBuffer(t.buffer, t.dtype, t.numel());
    const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    await writer.add(`tensors/${id}.bin`, new zip.Uint8ArrayReader(bytes));
  }

  // 5. Finalize
  await writer.close();
  return blobWriter.getData();
}

/**
 * Deserialize a ZIP Blob/File into a state dictionary.
 * Supports both torch.js custom JSON format (model.json) and PyTorch's pickle format (data.pkl).
 * For data.pkl, it unpickles the structure but does NOT reconstruct Tensors yet.
 */
export async function deserializeFromZip(blob: Blob): Promise<any> {
  // Open Zip
  const reader = new zip.ZipReader(new zip.BlobReader(blob));
  const entries = await reader.getEntries();

  // Try to find data.pkl (PyTorch format) first
  const pklEntry = entries.find(e => e.filename === 'data.pkl');
  const jsonEntry = entries.find(e => e.filename === MODEL_FILENAME);

  let revivedObj: any;

  if (pklEntry) {
    const pklUint8Data = await (pklEntry as any).getData(new zip.Uint8ArrayWriter());
    const parser = new Parser();
    const pickledObj = parser.parse(pklUint8Data);

    // This is the complex part: mapping PyTorch pickled objects to torch.js Tensors
    // PyTorch's data.pkl contains a pickled state_dict where Tensors are replaced
    // by instructions on how to rebuild them, often referencing binary files like `archive/data/X`.
    // The `pickledObj` here will contain a JavaScript representation of these instructions.
    // TODO: Implement a proper PyTorch tensor rebuilder for pickled state_dicts.
    //       This would involve traversing `pickledObj` to find tensor placeholders,
    //       reading corresponding binary data from zip entries (e.g., `archive/data/0`),
    //       and creating `torch.js` Tensor objects.
    console.warn("PyTorch-style data.pkl found. Unpickled object returned. Tensors require manual reconstruction via PyTorch's `_rebuild_tensor_v2` logic.");
    revivedObj = pickledObj;

  } else if (jsonEntry) {
    // Fallback to our custom JSON format
    const jsonStr = await (jsonEntry as any).getData(new zip.TextWriter());
    const jsonObj = JSON.parse(jsonStr);

    const tensorEntries = new Map<string, zip.Entry>();
    for (const entry of entries) {
      if (entry.filename.startsWith('tensors/') && entry.filename.endsWith('.bin')) {
        const id = entry.filename.slice(8, -4);
        tensorEntries.set(id, entry);
      }
    }

    async function reviveJson(obj: any): Promise<any> {
      if (obj && typeof obj === 'object') {
        if (obj.__tensor__ === true) {
          const meta = obj as TensorMetadata;
          const entry = tensorEntries.get(meta.id);
          if (!entry) {
            throw new Error(`Missing tensor data for id ${meta.id}`);
          }
          
          const uint8Data = await (entry as any).getData(new zip.Uint8ArrayWriter());
          
          const Constructor = getTypedArrayConstructor(meta.dtype);
          const bytesPerElement = getDTypeBytes(meta.dtype);
          
          const length = uint8Data.byteLength / bytesPerElement;
          
          const safeBuffer = new Uint8Array(uint8Data); 
          const typedData = new Constructor(safeBuffer.buffer, safeBuffer.byteOffset, length);
          
          const t = tensor(typedData as any, { dtype: meta.dtype }).reshape(meta.shape);
          if (meta.requires_grad) {
              t.requires_grad_(true);
          }
          return t;
        }
        
        if (Array.isArray(obj)) {
          return Promise.all(obj.map(reviveJson));
        } else {
          const newObj: any = {};
          for (const k in obj) {
            newObj[k] = await reviveJson(obj[k]);
          }
          return newObj;
        }
      }
      return obj;
    }
    revivedObj = await reviveJson(jsonObj);

  } else {
    throw new Error(`Invalid archive: neither ${MODEL_FILENAME} nor data.pkl found.`);
  }

  await reader.close();
  return revivedObj;
}