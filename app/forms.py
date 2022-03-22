from flask_wtf import FlaskForm
from wtforms import RadioField, SubmitField, HiddenField
from wtforms.validators import DataRequired

class LabelForm(FlaskForm):
    choice = RadioField(u'Label', choices=[('H', u'Healthy'), ('B', u'Unhealthy')], validators = [DataRequired(message='Cannot be empty')])
    submit = SubmitField('Add Label')
    corn_picture = HiddenField('default hidden field?')
