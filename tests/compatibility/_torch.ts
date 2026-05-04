import { beforeAll } from 'vitest';
import * as torch from '../../src/index';

beforeAll(async () => {
  await torch.init();
});

export { torch };
