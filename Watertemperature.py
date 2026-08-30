import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
from io import StringIO
import matplotlib

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

data = """Time(s)	Outlet water temperature (°C)
0.14 	100.369
1.89 	102.424
3.64 	106.39
5.39 	102.768
7.14 	101.369
8.88 	100.681
10.63 	102.017
12.38 	102.18
14.13 	102.515
15.88 	102.508
17.63 	104.582
19.38 	106.82
21.13 	102.894
22.87 	103.057
24.62 	102.374
26.37 	101.886
28.11 	100.682
29.86 	98.604
31.61 	95.326
33.36 	91.926
35.10 	89.443
36.85 	87.169
38.60 	85.561
40.34 	83.925
42.09 	82.552
43.84 	81.394
45.59 	80.359
47.34 	79.29
49.09 	78.472
50.84 	77.666"""

df = pd.read_csv(StringIO(data), sep='\t')

df.columns = df.columns.str.strip()

fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)

ax.set_xlim(df['Time(s)'].min() - 1, df['Time(s)'].max() + 1)
ax.set_ylim(df['Outlet water temperature (°C)'].min() - 5,
            df['Outlet water temperature (°C)'].max() + 5)


ax.set_xlabel('Time (s)', fontsize=18, fontname='Arial', labelpad=10)
ax.set_ylabel('Outlet temperature (°C)', fontsize=18, fontname='Arial', labelpad=10)

ax.set_title('Outlet temperature with Time', fontsize=18, fontweight='bold',
             fontname='Arial', pad=15)

ax.tick_params(axis='both', which='major', labelsize=18)

ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

line, = ax.plot([], [], '#F59092', linewidth=3, alpha=0.7)
points, = ax.plot([], [], 'ro', markersize=10)


temp_text = ax.text(0, 0, '', fontsize=24, fontname='Arial', color='red')


def init():
    line.set_data([], [])
    points.set_data([], [])
    temp_text.set_text('')
    return line, points, temp_text


def update(frame):
    x_data = df['Time(s)'].iloc[:frame + 1]
    y_data = df['Outlet water temperature (°C)'].iloc[:frame + 1]
    x_current = df['Time(s)'].iloc[frame]
    y_current = df['Outlet water temperature (°C)'].iloc[frame]
    line.set_data(x_data, y_data)


    if frame >= 0:
        points.set_data([x_current], [y_current])
        temp_text.set_text(f'{y_current:.1f}°C')
        temp_text.set_position((x_current + 0.8, y_current + 0.8))

    return line, points, temp_text


num_points = len(df)
total_duration = 50
interval_ms = (total_duration * 1000) / num_points

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(df),
    init_func=init,
    blit=False,
    interval=interval_ms,
    repeat=False
)


try:
    fps = num_points / total_duration
    ani.save('temperature_animation.gif', writer='pillow', fps=fps, dpi=100)
    print(f"GIF has been saved 'temperature_animation.gif'")
    print(f"total time: {total_duration}s")
    print(f"count: {num_points}point")
    print(f"interval: {interval_ms:.2f}ms")
    print(f"GIF frame rate: {fps:.2f} fps")
except Exception as e:
    print(f"An error occurred when saving the GIF: {e}")
    print("Please ensure pillow is installed: pip install pillow")

plt.tight_layout()
plt.show()