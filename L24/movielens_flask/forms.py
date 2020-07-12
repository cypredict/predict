#coding=utf-8
from flask_wtf import FlaskForm  # 表单相关
from wtforms import StringField, SubmitField, PasswordField, IntegerField  # 表单相关
from wtforms.validators import DataRequired  # 表单相关

#登录表单
class Login_Form(FlaskForm):
    username=StringField(u'用户名',validators=[DataRequired()])
    password=PasswordField(u'密 码',validators=[DataRequired()])
    submit=SubmitField(u'登录')

#注册表单
class Register_Form(FlaskForm):
    username=StringField(u'用户名',validators=[DataRequired(u"请避免出现空格")])
    password=PasswordField(u'密 码',validators=[DataRequired(u"请避免出现空格")])
    #email = StringField(u'邮 箱', validators=[DataRequired()])
    #num = StringField(u'验证码', validators=[DataRequired()])
    submit=SubmitField(u'注   册')
    #submit_right=SubmitField(u"发送验证码")

#查询表单
class Recommond_Form(FlaskForm):
    id=IntegerField(u'用户编号（1~950）',validators=[DataRequired()])
    submit=SubmitField(u'查看推荐')