export type SaveFunc = (obj: any, f?: string) => Promise<void>;
export type LoadFunc = (f: string | File | Blob) => Promise<any>;
