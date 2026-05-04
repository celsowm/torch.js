/**
 * PyTorch .pt file loader.
 * Parses zip archives with data.pkl and reconstructs tensors.
 * @status implemented
 * @pytorch torch.load()
 */

import { Parser, NameRegistry } from '../vendor/pickleparser/src/index';
import { Tensor } from '../tensor/Tensor';
import { tensor } from '../ops/creation';
import { DType } from '../dtype';
import * as zip from '../vendor/zip.js/index.js';

// PyTorch dtype mapping (only dtypes supported by torch.js)
const STORAGE_MAP: Record<string, { dtype: DType; bytesPerElement: number; TypedArray: new(b: ArrayBufferLike) => ArrayBufferView }> = {
  'FloatStorage': { dtype: 'float32' as DType, bytesPerElement: 4, TypedArray: Float32Array },
  'DoubleStorage': { dtype: 'float32' as DType, bytesPerElement: 8, TypedArray: Float64Array }, // Downcast to float32
  'HalfStorage': { dtype: 'float16' as DType, bytesPerElement: 2, TypedArray: Uint16Array },
  'BFloat16Storage': { dtype: 'float16' as DType, bytesPerElement: 2, TypedArray: Uint16Array },
  'LongStorage': { dtype: 'int32' as DType, bytesPerElement: 8, TypedArray: BigInt64Array }, // Downcast to int32
  'IntStorage': { dtype: 'int32' as DType, bytesPerElement: 4, TypedArray: Int32Array },
  'ShortStorage': { dtype: 'int32' as DType, bytesPerElement: 2, TypedArray: Int16Array },
  'CharStorage': { dtype: 'int8' as DType, bytesPerElement: 1, TypedArray: Int8Array },
  'ByteStorage': { dtype: 'uint8' as DType, bytesPerElement: 1, TypedArray: Uint8Array },
  'BoolStorage': { dtype: 'bool' as DType, bytesPerElement: 1, TypedArray: Uint8Array },
};

/**
 * Load a PyTorch .pt file from a ZIP Blob.
 * Returns the reconstructed state_dict or model weights.
 */
export async function loadPyTorchZip(blob: Blob): Promise<any> {
  const reader = new zip.ZipReader(new zip.BlobReader(blob));
  const entries = await reader.getEntries();

  // Find data.pkl
  const pklEntry = entries.find(e => e.filename === 'data.pkl');
  if (!pklEntry) {
    throw new Error('Not a valid PyTorch zip archive: data.pkl not found');
  }

  // Extract all data/<N> files (storage binaries)
  const storageData = new Map<number, Uint8Array>();
  for (const entry of entries) {
    const match = entry.filename.match(/^data\/(\d+)$/);
    if (match) {
      const idx = parseInt(match[1], 10);
      const uint8Data = await (entry as any).getData(new zip.Uint8ArrayWriter());
      storageData.set(idx, uint8Data);
    }
  }

  // Persistent resolver: maps storage ID to raw bytes
  const persistentResolver = {
    resolve: (persistentId: string): Uint8Array | null => {
      const num = parseInt(persistentId, 10);
      if (!isNaN(num)) {
        return storageData.get(num) || null;
      }
      try {
        const arr = JSON.parse(persistentId);
        if (Array.isArray(arr)) {
          const idx = arr.length >= 3 ? arr[2] : arr[0];
          return storageData.get(idx) || null;
        }
      } catch {}
      return null;
    },
  };

  // Storage cache: maps storage key -> TypedArray
  const storageCache = new Map<string, ArrayBufferView>();

  // Name registry for PyTorch functions
  const nameRegistry = new NameRegistry();

  // Register storage types
  for (const [storageName, config] of Object.entries(STORAGE_MAP)) {
    nameRegistry.register('torch', storageName, (persistentId: any) => {
      const key = JSON.stringify(persistentId);
      if (storageCache.has(key)) {
        return storageCache.get(key);
      }

      const rawBytes = persistentResolver.resolve(String(persistentId));
      if (!rawBytes) {
        throw new Error(`Storage not found for persistent ID: ${JSON.stringify(persistentId)}`);
      }

      const typedArray = new config.TypedArray(rawBytes.buffer.slice(rawBytes.byteOffset, rawBytes.byteOffset + rawBytes.byteLength));
      storageCache.set(key, typedArray);
      return typedArray;
    });
  }

  // Register _rebuild_tensor_v2
  nameRegistry.register('torch._utils', '_rebuild_tensor_v2', (
    storage: ArrayBufferView,
    storageOffset: number,
    size: number[],
    stride: number[],
    requiresGrad: boolean,
    _backwardHooks: any
  ) => {
    if (!storage) {
      throw new Error(`Invalid storage for _rebuild_tensor_v2`);
    }

    const numel = size.reduce((a, b) => a * b, 1);

    let dtype: DType = 'float32';
    if (storage instanceof Float32Array) dtype = 'float32';
    else if (storage instanceof Float64Array) dtype = 'float32'; // Downcast
    else if (storage instanceof Int32Array) dtype = 'int32';
    else if (storage instanceof Int16Array) dtype = 'int32'; // Upcast
    else if (storage instanceof Int8Array) dtype = 'int8';
    else if (storage instanceof Uint8Array) dtype = 'uint8';
    else if (storage instanceof BigInt64Array) dtype = 'int32'; // Downcast

    const startIdx = storageOffset;
    const data = storage.constructor === BigInt64Array
      ? Array.from((storage as BigInt64Array).slice(startIdx, startIdx + numel))
      : Array.from((storage as Float32Array).slice(startIdx, startIdx + numel));

    const t = tensor(data as any, { dtype });
    return t.reshape(size);
  });

  // Register _rebuild_parameter
  nameRegistry.register('torch._utils', '_rebuild_parameter', (
    t: Tensor,
    requiresGrad: boolean,
    _backwardHooks: any
  ) => {
    if (t instanceof Tensor) {
      t.requires_grad_(requiresGrad);
    }
    return t;
  });

  // Register OrderedDict
  nameRegistry.register('collections', 'OrderedDict', (...args: any[]) => {
    if (args.length === 1 && Array.isArray(args[0])) {
      const dict: Record<string, any> = {};
      for (const [key, value] of args[0]) {
        dict[key] = value;
      }
      return dict;
    }
    const dict: Record<string, any> = {};
    for (let i = 0; i < args.length; i += 2) {
      if (i + 1 < args.length) {
        dict[args[i]] = args[i + 1];
      }
    }
    return dict;
  });

  // Parse the pickle file
  const pklUint8Data = await (pklEntry as any).getData(new zip.Uint8ArrayWriter());
  const parser = new Parser({
    nameResolver: nameRegistry,
    persistentResolver: persistentResolver as any,
  });
  const result = parser.parse(pklUint8Data);

  await reader.close();
  return result;
}
