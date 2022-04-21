from flask_wtf import FlaskForm
from wtforms import RadioField, SubmitField, HiddenField
from wtforms.validators import DataRequired

class LabelForm(FlaskForm):
    h_choice = SubmitField(label='Healthy')
    b_choice = SubmitField(label='Unhealthy')
    corn_picture = HiddenField('default hidden field? idk how work')
