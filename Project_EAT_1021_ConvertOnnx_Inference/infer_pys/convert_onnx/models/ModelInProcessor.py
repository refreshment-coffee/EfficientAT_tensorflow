import multiprocessing as mp
import tensorflow as tf
import gc
import os
import psutil
import subprocess
from tensorflow.keras import backend as K

upLimit = 10000  # 每隔 upLimit 次推理，重启处理器

def runFirst():
    mp.set_start_method('spawn', force=True)


def model_inference_process(params_queue, result_queue, modelPredict):
    # print(f"[子进程] PID = {os.getpid()} 启动")
    while True:
        params = params_queue.get()  # 阻塞等待
        # print(f"当前子进程 PID = {os.getpid()} ")
        if any(p is None for p in params):  # 退出信号
            # print(f"[子进程] PID = {os.getpid()} 收到关闭信号，正在退出")
            break
        result = modelPredict(*params)
        result_queue.put(result)


class ModelInfProcessor:
    def __init__(self, modelPredict):
        self.modelPredict = modelPredict
        self.params_queue = mp.Queue(maxsize=1)
        self.result_queue = mp.Queue(maxsize=1)
        self.process = mp.Process(
            target=model_inference_process,
            args=(self.params_queue, self.result_queue, modelPredict)
        )
        self.process.start()
        self.inf_counter = 0

    def predict(self, *args):
        params = []
        for p in args:
            if isinstance(p, tf.Tensor):
                p = p.numpy()
            params.append(p)
        self.params_queue.put(params)
        self.inf_counter += 1
        return self.result_queue.get()

    def should_restart(self):
        return self.inf_counter % upLimit == 0

    def shutdown(self):
        # print(f"正在尝试关闭子进程 PID = {self.process.pid}")
        try:
            self.params_queue.put([None])  # 通知子进程退出
            self.process.join(timeout=2)
            self.process.terminate()       # 代码方面强制退出
            self.process.join(timeout=2)
            # print("状态： ",psutil.Process(self.process.pid).status())

            ps_proc = psutil.Process(self.process.pid)  ##通过进程ID获取进程信息
            status = ps_proc.status()
            # print(f"子进程 PID = {self.process.pid} 状态: {status}")

            if status in [psutil.STATUS_SLEEPING, psutil.STATUS_IDLE]:  ##看看需不需要zombie----psutile.STATUS_ZOMBIE
                # print("子进程 sleeping ，重新使用该进程")
                return True  # 表示复用

            subprocess.run(["kill", "-9", str(self.process.pid)])
            # print("强制 kill 子进程成功")

        # except Exception as e:
            # print(f"shutdown函数 异常: {e}")
        finally:
            # print(self.process == None)
            # print(type(self.process))
            if self.process.is_alive():
                self.process.terminate()
            self.process.close()
            self.process = None
            K.clear_session()
            gc.collect()
            return False  # 表示已杀死，需要新建


def getProcessor(current_processor: ModelInfProcessor, modelPredict):
    """
    包装函数，管理 processor 生命周期。推荐外部传入 current_processor 对象。
    如果达到 upLimit 次数，会优先尝试复用子进程，否则重建。
    """
    if current_processor is None:
        return ModelInfProcessor(modelPredict)

    if current_processor.should_restart():
        reused = current_processor.shutdown()
        if reused:
            return current_processor
        else:
            return ModelInfProcessor(modelPredict)

    return current_processor
