# parse_args()详解

---

初步设置

> echo 前加 -- 变成 --echo 使得这个参数变为可选参数

```python
# 1引入模块
import argparse

# 2建立解析对象
parser = argparse.ArgumentParser()

# 3增加属性：给xx实例增加一个aa属性 # xx.add_argument("aa")
parser.add_argument("--echo",default="You dont enter this argument")

# 4属性给与args实例： 把parser中设置的所有"add_argument"给返回到args子类实例当中， 那么parser中增加的属性内容都会在args实例中，使用即可。
args = parser.parse_args()
parser.parse_args()

# 打印定位参数echo
print(args.echo)
```

运行结果

![image-20220721133346273](C:\Users\zhuozhuo\AppData\Roaming\Typora\typora-user-images\image-20220721133346273.png)

---

设置help：

```python
# 1引入模块
import argparse

# 2建立解析对象
parser = argparse.ArgumentParser()

# 3增加属性：给xx实例增加一个aa属性 # xx.add_argument("aa") # 可以设置help信息
parser.add_argument("--echo",help = "echo is here", default="You dont enter this argument")

# 4属性给与args实例： 把parser中设置的所有"add_argument"给返回到args子类实例当中， 那么parser中增加的属性内容都会在args实例中，使用即可。
args = parser.parse_args()
parser.parse_args()

# 打印定位参数echo
print(args.echo)
```

运行结果

![image-20220721133704504](C:\Users\zhuozhuo\AppData\Roaming\Typora\typora-user-images\image-20220721133704504.png)

---

