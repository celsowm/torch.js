import torch from '../src';
import { getCapabilities } from '../src/backend/webgpu/capabilities';

await torch.init();

const caps = getCapabilities();
console.log('🔥 Sua GPU:', caps.platform.gpu);
console.log('🌐 Browser:', caps.platform.browser);
console.log('💾 Shared Memory:', caps.workgroupSharedMemory);
console.log('📏 Limits:', caps.limits);
