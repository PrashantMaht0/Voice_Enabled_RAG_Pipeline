// Request and reply over a worker: each call gets an id; progress messages for that id go to its callback.
export type Progress = { step: string; value: number };

export class WorkerRpc {
  private next = 1;
  private pending = new Map<number, { resolve: (v: any) => void; reject: (e: Error) => void; onProgress?: (p: Progress) => void }>();

  constructor(readonly worker: Worker) {
    worker.onmessage = (event: MessageEvent) => {
      const { id, result, error, progress } = event.data;
      const call = this.pending.get(id);
      if (!call) return;
      if (progress) { call.onProgress?.(progress); return; }
      this.pending.delete(id);
      if (error) call.reject(new Error(error)); else call.resolve(result);
    };
    worker.onerror = (event) => {
      for (const call of this.pending.values()) call.reject(new Error(event.message || "worker failed"));
      this.pending.clear();
    };
  }

  call<T>(op: string, args: object = {}, onProgress?: (p: Progress) => void, transfer: Transferable[] = []): Promise<T> {
    const id = this.next++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onProgress });
      this.worker.postMessage({ id, op, args }, transfer);
    });
  }
}
