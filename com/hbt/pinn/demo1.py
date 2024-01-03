import time

start_time = time.time()
end_time = time.time()
print("训练共用时" + str(end_time - start_time) + "秒")


def ab_from_points(x1, y1, x2, y2):
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return a, b


# print("请输入第一个点的坐标：")
# x1 = float(input("请输入第一个点x坐标："))
# y1 = float(input("请输入第一个点y坐标："))
# h1 = (x1, y1)
# x2 = float(input("请输入第二个点x坐标："))
# y2 = float(input("请输入第二个点y坐标："))
# h2 = (x2, y2)
# 从键盘输入获取二维坐标点
# point_str = input("请输入第一个h1坐标点（格式为 x,y）：")
# # 将输入的字符串分割为两个值
# x1, y1 = map(float, point_str.split(','))
# point_str = input("请输入第二个h2坐标点（格式为 x,y）：")
# # 将输入的字符串分割为两个值
# x2, y2 = map(float, point_str.split(','))
# a, b = ab_from_points(x1,y1,x2,y2)
# print(a, b)
# print("这条直线为：y=" + str(a) + "x+" + str(b))
x1 = 20
x2 = 0
for h1 in range(100):
    a,b=ab_from_points(x1, h1 + 1, x2, h1 + 101)
    print("这条直线为：y=" + str(a) + "x+" + str(b))
