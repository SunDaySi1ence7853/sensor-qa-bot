# 工具层接通数据源演示(第八阶段任务1)
# 用法:先在 config.yaml 切 data_source，再运行本脚本，对比两次输出:
#   1) data_source: simulated → python demo_tools_datasource.py
#   2) data_source: serial(接好板子) → python demo_tools_datasource.py
# 输出里的 data_source 字段就是"模拟↔真实"的铁证
import time

from src.tools.sensor_tools import get_sensor_data, query_history


def main():
    print("== 实时读数 ×5 ==")
    for _ in range(5):
        print(get_sensor_data.invoke({"sensor_type": "temperature"}))
        time.sleep(2)

    print("\n== 湿度(串口模式下若硬件未接会如实报错) ==")
    print(get_sensor_data.invoke({"sensor_type": "humidity"}))

    print("\n== 历史摘要 ==")
    print(query_history.invoke({"sensor_type": "temperature", "hours": 1}))


if __name__ == "__main__":
    main()