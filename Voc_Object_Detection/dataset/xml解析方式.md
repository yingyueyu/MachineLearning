# [Python-解析 XML](https://www.cnblogs.com/qjj19931230/p/12555038.html)

## 简介

XML 全称 Extensible Markup Language，中文译为可扩展标记语言。

XML 之前有两个先行者：SGML 和 HTML，率先登场的是 SGML， 尽管它功能强大，但文档结构复杂，既不容易学也不易于使用，因此几个主要的浏览器厂商均拒绝支持 SGML，这些因素限制了 SGML 在网上的传播性；1989 年 HTML 登场，它继承了 SGML 诸多优点，去除了 SGML 复杂庞大的缺点，HTML 在数据显示上表现十分出色，但它的语法是不可扩展的，因此其无法描述数据、可读性差，没办法人们再次将目光转向 SGML，经过对 SGML 一系列改造，终于在 1998 年，XML 第一个版本问世。

简单来说就是：XML 和 HTML 均由 SGML 改造而来，HTML 是一种页面技术，聚焦的是数据的显示，而 XML 易于扩展，主要用来传送和存储数据，聚焦的是数据的内容。

## 解析

### 解析方式

Python 有三种 XML 解析方式：SAX（simple API for XML）、DOM（Document Object Model）、ElementTree。

- DOM 方式：DOM 中文译为文档对象模型，是 W3C 组织推荐的标准编程接口，它将 XML 数据在内存中解析成一个树，通过对树的操作来操作 XML。
- SAX 方式：SAX 是一个用于处理 XML 事件驱动的模型，它逐行扫描文档，一边扫描一边解析，对于大型文档的解析拥有巨大优势，尽管不是 W3C 标准，但它却得到了广泛认可。
- ElementTree 方式：ElementTree 相对于 DOM 来说拥有更好的性能，与 SAX 性能差不多，API 使用也很方便。

### 具体实现

在具体解析之前我们先准备一个 XML，如下所示：

test.xml

```
<?xml version="1.0" encoding="utf-8"?>
<list>
<student id="stu1" name="stu">
   <id>1001</id>
   <name>张三</name>
   <age>22</age>
   <gender>男</gender>
</student>
<student id="stu2" name="stu">
   <id>1002</id>
   <name>李四</name>
   <age>21</age>
   <gender>女</gender>
</student>
</list>
```

#### DOM 方式解析

使用 DOM 方式，首先要对其 API 有一定了解，如果不了解，网上的教程也比较多，比如：DOM 教程，下面看一下使用示例。

```
from xml.dom.minidom import parse

# 读取文件
dom = parse('test.xml')
# 获取文档元素对象
data = dom.documentElement
# 获取 student
stus = data.getElementsByTagName('student')
for stu in stus:
    # 获取标签属性值
    st_id = stu.getAttribute('id')
    st_name = stu.getAttribute('name')
    # 获取标签中内容
    id = stu.getElementsByTagName('id')[0].childNodes[0].nodeValue
    name = stu.getElementsByTagName('name')[0].childNodes[0].nodeValue
    age = stu.getElementsByTagName('age')[0].childNodes[0].nodeValue
    gender = stu.getElementsByTagName('gender')[0].childNodes[0].nodeValue
    print('st_id:', st_id,  ', st_name:',st_name)
    print('id:', id, ', name:', name, ', age:', age, ', gender:',gender)
```

输出结果：

```
st_id: stu1 , st_name: stu
id: 1001 , name: 张三 , age: 22 , gender: 男
st_id: stu2 , st_name: stu
id: 1002 , name: 李四 , age: 21 , gender: 女
```



 

通过输出结果，我们可以发现已经获取了标签属性值和标签内容了。

#### SAX 方式解析

使用 SAX 解析 XML 文档主要涉及到解析器和事件处理器，解析器负责读取 XML 文档，并向事件处理器发送事件，事件处理器负责对事件作出响应，对传递的 XML 数据进行处理。

Python 使用 SAX 处理 XML 需要用到 xml.sax 中的 parse 函数和 xml.sax.handler 中的 ContentHandler 类，下面看一下 ContentHandler 类中的一些方法。

- characters(content)：调用时机：从行开始，遇到标签之前，存在字符，content 的值为这些字符串；从一个标签，遇到下一个标签之前， 存在字符，content 的值为这些字符串；从一个标签，遇到行结束符之前，存在字符，content 的值为这些字符串。
- startDocument()：文档启动的时候调用。
- endDocument()：解析器到达文档结尾时调用。
- startElement(name, attrs)：遇到 XML 开始标签时调用，name 是标签的名字，attrs 是标签的属性值字典。
- endElement(name)：遇到 XML 结束标签时调用。

下面通过示例看一下如何通过 SAX 方式解析 XML。

```
import xml.sax

class StudentHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.id = ""
        self.name = ""
        self.age = ""
        self.gender = ""

    # 元素开始调用
    def startElement(self, tag, attributes):
        self.CurrentData = tag
        if tag == "student":
            stu_name = attributes["name"]
            print("stu_name:", stu_name)

    # 元素结束调用
    def endElement(self, tag):
        if self.CurrentData == "id":
            print("id:", self.id)
        elif self.CurrentData == "name":
            print("name:", self.name)
        elif self.CurrentData == "age":
            print("age:", self.age)
        elif self.CurrentData == "gender":
            print("gender:", self.gender)
        self.CurrentData = ""

    # 读取字符时调用
    def characters(self, content):
        if self.CurrentData == "id":
            self.id = content
        elif self.CurrentData == "name":
            self.name = content
        elif self.CurrentData == "age":
            self.age = content
        elif self.CurrentData == "gender":
            self.gender = content

if (__name__ == "__main__"):
    # 创建 XMLReader
    parser = xml.sax.make_parser()
    # 关闭命名空间
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    # 重写 ContextHandler
    Handler = StudentHandler()
    parser.setContentHandler(Handler)
    parser.parse("test.xml")
```

 

输出结果：

```
stu_name: stu
id: 1001
name: 张三
age: 22
gender: 男
stu_name: stu
id: 1002
name: 李四
age: 21
gender: 女
```

 

#### ElementTree 方式解析

Python 提供了两种 ElementTree 的实现方式。一个是纯 Python 实现的 xml.etree.ElementTree，另一个是 C 语言实现 xml.etree.cElementTree，使用 C 语言实现的方式速度更快且内存消耗更少。Python3.3 之后，ElemenTree 模块会自动优先使用 C 加速器，如果不存在 C 实现，则会使用 Python 实现。因此，使用 Python3.3+ 时，只需要 import xml.etree.ElementTree 即可。下面看一下示例。

```
import xml.etree.ElementTree as ET

tree = ET.parse("test.xml")
# 根节点
root = tree.getroot()
# 标签名
print('root_tag:',root.tag)
for stu in root:
    # 属性值
    print ("stu_name:", stu.attrib["name"])
    # 标签中内容
    print ("id:", stu[0].text)
    print ("name:", stu[1].text)
    print("age:", stu[2].text)
    print("gender:", stu[3].text)
```



输出结果：                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     

```
root_tag: list
stu_name: stu
id: 1001
name: 张三
age: 22
gender: 男
stu_name: stu
id: 1002
name: 李四
age: 21
gender: 女
```