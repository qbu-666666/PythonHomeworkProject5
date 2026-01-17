from django import template

register = template.Library()

@register.filter
def multiply(value, arg):
    """将值乘以一个数"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        try:
            return float(value) * arg
        except:
            return 0

@register.filter
def percentage(value):
    """将小数转换为百分比字符串"""
    try:
        return f"{float(value)*100:.1f}%"
    except (ValueError, TypeError):
        return "0%"