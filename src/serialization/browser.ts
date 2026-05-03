import { serializeToZip, deserializeFromZip } from './common';

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

export async function load(f: string | File | Blob): Promise<any> {
  if (typeof f === 'string') {
    const response = await fetch(f);
    if (!response.ok) throw new Error(`Failed to fetch ${f}`);
    const blob = await response.blob();
    return deserializeFromZip(blob);
  } else {
    return deserializeFromZip(f);
  }
}
