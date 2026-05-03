/**
 * Serialize a state dictionary (or any object containing tensors) to a ZIP Blob.
 * This currently uses a custom JSON-based format within the zip.
 *
 * TODO: For PyTorch compatibility, this should ideally create a `data.pkl` file
 *       using a Python-compatible pickler, replacing Tensors with references to
 *       binary data files (e.g., `archive/data/X`). This requires a TypeScript
 *       pickler implementation.
 */
declare function serializeToZip(obj: any): Promise<Blob>;
/**
 * Deserialize a ZIP Blob/File into a state dictionary.
 * Supports both torch.js custom JSON format (model.json) and PyTorch's pickle format (data.pkl).
 * For data.pkl, it unpickles the structure but does NOT reconstruct Tensors yet.
 */
declare function deserializeFromZip(blob: Blob): Promise<any>;

export { deserializeFromZip, serializeToZip };
