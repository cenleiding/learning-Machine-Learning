# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2019/1/23
@description: 练习
"""
import matplotlib.pyplot as plt
import matplotlib.style as style

print(style.available)
print(plt.__file__)
style.use('bmh')
x = [1,2,3,4,5]
y = [1,2,3,4,5]
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax1.plot(x,y,label='sb',color='r')
ax1.fill_between(x,y,0)
ax1.set_xlabel('www')
ax1.legend()
# plt.plot(x,y,label='sb',color='r')
# plt.bar([1,2,3,4,5],[1,2,3,4,5],label='sb',color='r')
# plt.scatter(x,y,label='sb',color='r',s=25,marker='o')
# plt.fill_between(x,y,0)
# plt.xlabel('s',color='r')
# plt.ylabel('b')
# plt.legend()
# plt.title('lalala')



plt.show()
