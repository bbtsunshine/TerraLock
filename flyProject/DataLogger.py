import pandas as pd
import json
import time
from pathlib import Path
from datetime import datetime
import threading
import inspect
import sys
import os


class DataLogger:
    def __init__(self, base_filename="experiment_data", auto_save_interval=100):
        """初始化记录器：文件名+自动保存间隔"""
        self.data = {}
        self.metadata = {
            "start_time": datetime.now().isoformat(),
            "variables": {},
            "parameters": {},  # 存储运行时参数
            "system_info": self._get_system_info()  # 系统信息
        }
        self.base_filename = base_filename
        self.auto_save_interval = auto_save_interval
        self.counter = 0
        self.records = []
        self.excel_file_path = f"data_logs/{base_filename}.xlsx"
        Path("data_logs").mkdir(exist_ok=True)  # 创建存储目录

        # 初始化Excel文件（如果已存在则保留）
        if not os.path.exists(self.excel_file_path):
            with pd.ExcelWriter(self.excel_file_path, engine='openpyxl') as writer:
                # 创建空的工作表
                pd.DataFrame().to_excel(writer, sheet_name="Data", index=False)
                pd.DataFrame(columns=["Variable", "Description"]).to_excel(
                    writer, sheet_name="Variables", index=False)
                pd.DataFrame(columns=["Parameter", "Value"]).to_excel(
                    writer, sheet_name="Parameters", index=False)
                pd.DataFrame(columns=["Property", "Value"]).to_excel(
                    writer, sheet_name="System Info", index=False)

    def _get_system_info(self):
        """获取系统信息"""
        import platform
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "system_time": datetime.now().isoformat()
        }

    def register_variable(self, name: str, description: str = ""):
        """注册需记录的变量（如未注册，log时自动注册）"""
        if name not in self.data:
            self.data[name] = []
            self.metadata["variables"][name] = description

    def log_parameters(self, **kwargs):
        """
        记录实验参数/配置
        这些参数在整个实验过程中保持不变
        """
        for name, value in kwargs.items():
            # 将参数转换为可JSON序列化的格式
            if hasattr(value, '__dict__'):  # 如果是对象，尝试转换为字典
                try:
                    value = vars(value)
                except:
                    value = str(value)
            elif callable(value):  # 如果是函数，记录其名称
                value = value.__name__

            self.metadata["parameters"][name] = value

        # 立即保存参数到Excel文件
        self._update_excel_parameters()

    def log_variables(self, **kwargs):
        """记录当前变量值（支持动态注册）"""
        timestamp = time.time()
        record = {"timestamp": timestamp, "iteration": self.counter}

        for name, value in kwargs.items():
            if name not in self.data:  # 自动注册未定义的变量
                self.register_variable(name, f"Auto-registered at {datetime.now().isoformat()}")
            self.data[name].append(value)
            record[name] = value

        self.records.append(record)
        self.counter += 1

        # 自动保存检查
        if self.counter % self.auto_save_interval == 0:
            self.save_data()

    def _update_excel_parameters(self):
        """更新Excel文件中的参数工作表"""
        # 读取现有的Excel文件
        with pd.ExcelWriter(self.excel_file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            # 更新参数工作表
            pd.DataFrame(
                self.metadata["parameters"].items(),
                columns=["Parameter", "Value"]
            ).to_excel(writer, sheet_name="Parameters", index=False)

    def save_data(self, format: str = "all"):
        """保存数据到文件（CSV/JSON/Excel）"""
        self.metadata["end_time"] = datetime.now().isoformat()
        self.metadata["total_iterations"] = self.counter

        df = pd.DataFrame(self.records)

        if format in ["csv", "all"]:
            csv_path = f"data_logs/{self.base_filename}.csv"
            df.to_csv(csv_path, index=False)

        if format in ["json", "all"]:
            json_path = f"data_logs/{self.base_filename}.json"
            with open(json_path, "w") as f:
                json.dump({"metadata": self.metadata, "data": self.records}, f, indent=2)

        if format in ["excel", "all"]:
            # 使用mode='a'以追加模式打开，if_sheet_exists='replace'替换现有工作表
            with pd.ExcelWriter(self.excel_file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                # 数据工作表
                df.to_excel(writer, sheet_name="Data", index=False)

                # 变量描述工作表
                pd.DataFrame(
                    self.metadata["variables"].items(),
                    columns=["Variable", "Description"]
                ).to_excel(writer, sheet_name="Variables", index=False)

                # 参数工作表
                pd.DataFrame(
                    self.metadata["parameters"].items(),
                    columns=["Parameter", "Value"]
                ).to_excel(writer, sheet_name="Parameters", index=False)

                # 系统信息工作表
                pd.DataFrame(
                    self.metadata["system_info"].items(),
                    columns=["Property", "Value"]
                ).to_excel(writer, sheet_name="System Info", index=False)

    def get_parameter_summary(self):
        """返回参数字符串摘要，可用于日志记录"""
        summary = "实验参数:\n"
        for param, value in self.metadata["parameters"].items():
            summary += f"  {param}: {value}\n"
        return summary


# 使用示例
if __name__ == "__main__":
    # 初始化数据记录器
    logger = DataLogger(base_filename="model_training", auto_save_interval=10)

    # 记录实验参数
    logger.log_parameters(
        model_name="ResNet50",
        learning_rate=0.001,
        batch_size=64,
        epochs=100,
        optimizer="Adam",
        loss_function="CrossEntropy",
        dataset="CIFAR-10",
        augmentation=True,
        comment="基线实验"
    )

    # 打印参数摘要
    print(logger.get_parameter_summary())

    # 注册要记录的变量
    logger.register_variable("loss", "Training loss")
    logger.register_variable("accuracy", "Model accuracy")
    logger.register_variable("learning_rate", "Current learning rate")

    # 模拟训练循环
    for epoch in range(100):
        loss = 1.0 / (epoch + 1)
        accuracy = 0.8 + epoch * 0.002
        lr = 0.001 * (0.95 ** epoch)
        logger.log_variables(loss=loss, accuracy=accuracy, learning_rate=lr)

    # 保存最终数据
    logger.save_data("all")