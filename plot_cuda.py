import matplotlib.pyplot as plt
import os

os.makedirs('img', exist_ok=True)

data = {
    'N':      [256, 256, 256, 512, 512, 512, 1024, 1024, 1024, 2048, 2048, 2048],
    'Block':  [8, 16, 16, 8, 16, 16, 8, 16, 16, 8, 16, 16],
    'Mode':   ['tiled', 'tiled', 'naive', 'tiled', 'tiled', 'naive', 'tiled', 'tiled', 'naive', 'tiled', 'tiled', 'naive'],
    'Time':   [0.3652, 0.2387, 0.6060, 2.8631, 1.7082, 4.6105, 23.0026, 13.2967, 36.2861, 181.5614, 106.0933, 292.3870]
}

data_32 = {
    'N': [256, 512, 1024, 2048],
    'Time': [0.2281, 1.5937, 12.7107, 99.1625]
}

plt.style.use('seaborn-v0_8-whitegrid')

plt.figure(figsize=(10, 6))
for mode in ['tiled', 'naive']:
    ns = [d[0] for d in zip(data['N'], data['Block'], data['Mode'], data['Time']) if d[1] == 16 and d[2] == mode]
    ts = [d[3] for d in zip(data['N'], data['Block'], data['Mode'], data['Time']) if d[1] == 16 and d[2] == mode]
    plt.plot(ns, ts, marker='o', linewidth=2, label=mode)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Размер матрицы N', fontsize=12)
plt.ylabel('Время выполнения, мс', fontsize=12)
plt.title('Сравнение: Наивное ядро vs Оптимизированное (Tiled)', fontsize=14)
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig('img/naive_vs_tiled.png', dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(data_32['N'], data_32['Time'], marker='s', linewidth=2, label='Блок 32x32', color='green')
ns_16 = [d[0] for d in zip(data['N'], data['Block'], data['Mode'], data['Time']) if d[1] == 16 and d[2] == 'tiled']
ts_16 = [d[3] for d in zip(data['N'], data['Block'], data['Mode'], data['Time']) if d[1] == 16 and d[2] == 'tiled']
ns_8 = [d[0] for d in zip(data['N'], data['Block'], data['Mode'], data['Time']) if d[1] == 8 and d[2] == 'tiled']
ts_8 = [d[3] for d in zip(data['N'], data['Block'], data['Mode'], data['Time']) if d[1] == 8 and d[2] == 'tiled']

plt.plot(ns_16, ts_16, marker='o', linewidth=2, label='Блок 16x16', color='blue')
plt.plot(ns_8, ts_8, marker='^', linewidth=2, label='Блок 8x8', color='red')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Размер матрицы N', fontsize=12)
plt.ylabel('Время выполнения, мс', fontsize=12)
plt.title('Влияние размера блока на производительность (Tiled)', fontsize=14)
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig('img/block_sizes.png', dpi=300)
plt.close()

print("Графики для CUDA созданы в папке img!")