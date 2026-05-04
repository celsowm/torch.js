import { serializeToZip, deserializeFromZip, loadPyTorchZip } from './common';

export async function save(obj: any, filename?: string): Promise<void> {
  const blob = await serializeToZip(obj);

  if (typeof window !== 'undefined' && typeof document !== 'undefined') {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename || 'model.zip';
    a.click();
    URL.revokeObjectURL(url);
  } else {
    throw new Error("Browser environment required for save() without a filesystem polyfill.");
  }
}

/**
 * Load a serialized model file.
 * Supports:
 * - torch.js custom zip format (model.json + tensors/)
 * - PyTorch .pt/.pth zip format (data.pkl + data/0, data/1, ...)
 *
 * Automatically detects format and uses the appropriate loader.
 * @pytorch torch.load()
 */
export async function load(f: string | File | Blob): Promise<any> {
  let blob: Blob;
  if (typeof f === 'string') {
    const response = await fetch(f);
    if (!response.ok) throw new Error(`Failed to fetch ${f}`);
    blob = await response.blob();
  } else {
    blob = f;
  }

  // Detect PyTorch format: zip with data.pkl
  try {
    return await loadPyTorchZip(blob);
  } catch (e) {
    // If it's not a PyTorch zip, fall back to torch.js custom format
    return deserializeFromZip(blob);
  }
}

// Re-export loadPyTorchZip for explicit PyTorch loading
export { loadPyTorchZip };
