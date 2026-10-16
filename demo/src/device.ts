// Which answer model to suggest (spec 0009, pinned in build task 1): 1B on phones, small memory or small GPU buffers,
// else 3B; the f16 build needs the shader-f16 GPU feature. WebGPU exposes no total memory, so an out of memory
// error while loading 3B still offers 1B.
export type DeviceInfo = { webgpu: boolean; f16: boolean; mobile: boolean; suggested: "3b" | "1b"; reason: string };

const GIB = 1024 ** 3;

export async function inspectDevice(): Promise<DeviceInfo> {
  const gpu = (navigator as any).gpu;
  const adapter = gpu ? await gpu.requestAdapter().catch(() => null) : null;
  const mobile = (navigator as any).userAgentData?.mobile === true || /Android|iPhone|iPad/i.test(navigator.userAgent);
  if (!adapter) return { webgpu: false, f16: false, mobile, suggested: "1b", reason: "no WebGPU" };
  const memory: number | undefined = (navigator as any).deviceMemory;
  const smallBuffers = adapter.limits.maxBufferSize < 2 * GIB;
  const reasons = [mobile && "a phone or tablet", memory !== undefined && memory <= 4 && `${memory} GB of memory`,
    smallBuffers && "a GPU with small buffers"].filter(Boolean) as string[];
  return {
    webgpu: true,
    f16: adapter.features.has("shader-f16"),
    mobile,
    suggested: reasons.length ? "1b" : "3b",
    reason: reasons.length ? `suggested for ${reasons.join(" and ")}` : "suggested for this computer",
  };
}
