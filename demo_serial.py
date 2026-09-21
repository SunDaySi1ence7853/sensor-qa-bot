# 切换演示(第八阶段任务1)
# 用法: 
# 1. config.yaml 里 data_source: simulated 跑一次，截一张图
# 2. config.yaml 里 data_source: serial (插着板子) 跑一次，截一张图
import time
from src.tools.sensor_tools import get_sensor_data, query_history

def main():
    print("== 实时读数 ×5 ==")
    for _ in range(5):
        print(get_sensor_data.invoke({"sensor_type": "temperature"}))
        time.sleep(2)

    print("\n== 湿度(板子发 humi 帧就有值，没发会报错) ==")
    print(get_sensor_data.invoke({"sensor_type": "humidity"}))

    print("\n== 历史摘要 ==")
    print(query_history.invoke({"sensor_type": "temperature", "hours": 1}))

if __name__ == "__main__":
    main()