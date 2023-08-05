import numpy as np
import matplotlib.pyplot as plt
from fealpy.decorator import cartesian
from fealpy.mesh import UniformMesh1d
from scipy.sparse.linalg import spsolve
from fealpy.pde.wave_1d import StringOscillationSinCosPDEData

# PDE 模型
pde = StringOscillationSinCosPDEData(D=[0, 1], T=[0, 2])

# 空间离散
domain = pde.domain()
nx = 10
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])

# 时间离散
duration = pde.duration()
nt = 20
tau = (duration[1] - duration[0])/nt

# 准备初值
uh0 = mesh.interpolate(pde.init_solution, 'node')
vh0 = mesh.interpolate(pde.init_solution_diff_t, 'node')
uh1 = mesh.function('node')
# 时间步进
def advance(n, *frags):
    """
    @brief 时间步进

    @param[in] n int, 表示第 `n` 个时间步（当前时间步） 
    """
    theta = 0.5
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    elif n == 1:
        r = tau/hx
        uh1[1:-1] = r**2*(uh0[0:-2] + uh0[2:])/2.0 + (1-r**2)*uh0[1:-1] + tau*vh0[1:-1]
        gD = lambda p: pde.dirichlet(p, t)
        mesh.update_dirichlet_bc(gD, uh1)
        return uh1, t
    else:
        A0, A1, A2 = mesh.wave_operator(tau, theta=theta)
        source = lambda p: pde.source(p, t)
        f = mesh.interpolate(source, intertype='node')
        f *= tau**2
        f += A1@uh1 + A2@uh0

        uh0[:] = uh1
        gD = lambda p: pde.dirichlet(p, t)
        if theta == 0.0:
            uh1[:] = f
            mesh.update_dirichlet_bc(gD, uh1)
        else:
            A0, f = mesh.apply_dirichlet_bc(gD, A0, f)
            uh1[:] = spsolve(A0, f)

        return uh1, t
# 制作动画
box = [0, 1, -1.2, 1.2]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, advance, frames=nt+1)
plt.show()

# 获取特定时间层图像
fig, axs = plt.subplots(2, 2)  # 创建一个2x2的子图布局

times_to_plot = [0.5, 1.0, 1.5, 2.0]  # 需要绘制的时间层

x = mesh.entity("node")

#各时间层解的图像
for n in range(nt + 1):
    t = duration[0] + n * tau
    y, _ = advance(n)
    if t in times_to_plot:

        if np.isclose(t, times_to_plot[0]):
            def f1(x):
                return pde.solution(x, t=times_to_plot[0])

            axs[0, 0].plot(x, y)  # 第一个子图
        elif np.isclose(t, times_to_plot[1]):
            def f1(x):
                return pde.solution(x, t=times_to_plot[1])


            axs[0, 1].plot(x, y)  # 第二个子图
        elif np.isclose(t, times_to_plot[2]):
            def f1(x):
                return pde.solution(x, t=times_to_plot[2])

            axs[1, 0].plot(x, y)  # 第三个子图
        elif np.isclose(t, times_to_plot[3]):
            def f1(x):
                return pde.solution(x, t=times_to_plot[3])

            axs[1, 1].plot(x, y)  # 第四个子图
# 设置子图标题
axs[0, 0].set_title(f't = {times_to_plot[0]}')
axs[0, 1].set_title(f't = {times_to_plot[1]}')
axs[1, 0].set_title(f't = {times_to_plot[2]}')
axs[1, 1].set_title(f't = {times_to_plot[3]}')

# 调整子图布局
plt.tight_layout()

# 显示图形
plt.show()
# 计算误差
for n in range(nt + 1):
    t = duration[0] + n * tau
    y, _ = advance(n)

    if t in times_to_plot:
        if np.isclose(t, times_to_plot[0]):
            def f1(x):
                return pde.solution(x, t=times_to_plot[0])
            print(f"时间层为{times_to_plot[0]}的误差是:", mesh.error(f1, y))

        elif np.isclose(t, times_to_plot[1]):
            def f1(x):
                return pde.solution(x, t=times_to_plot[1])

            print(f"时间层为{times_to_plot[1]}的误差是:", mesh.error(f1, y))

        elif np.isclose(t, times_to_plot[2]):
            def f1(x):
                return pde.solution(x, t=times_to_plot[2])

            print(f"时间层为{times_to_plot[2]}的误差是:", mesh.error(f1, y))

        elif np.isclose(t, times_to_plot[3]):
            def f1(x):
                return pde.solution(x, t=times_to_plot[3])

            print(f"时间层为{times_to_plot[3]}的误差是:", mesh.error(f1, y))

