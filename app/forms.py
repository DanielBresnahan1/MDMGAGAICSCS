from flask_wtf import FlaskForm
from wtforms import RadioField, SubmitField, HiddenField
from wtforms.validators import DataRequired

class LabelForm(FlaskForm):
    # choice = RadioField(u'Label', choices=[('H', u'Healthy'), ('B', u'Unhealthy')], validators = [DataRequired(message='Cannot be empty')])
    # submit = SubmitField('Add Label')

    h_choice = SubmitField(label='Healthy')
    b_choice = SubmitField(label='Unhealthy')
    corn_picture = HiddenField('default hidden field? idk how work')
    # how to style flask-wtf forms?
    # use multiple submit buttons instead of gross radio buttons
    # https://stackoverflow.com/questions/43811779/use-many-submit-buttons-in-the-same-form
    # https://stackoverflow.com/questions/65938462/multiple-submit-buttons-in-flask?msclkid=3be99d9fb9fd11ecb482bf1760d889da
    # https://stackoverflow.com/questions/36090695/flask-wtforms-how-to-make-a-form-with-multiple-submit-buttons?msclkid=7e8cf03db9fe11ec977f0e7b11b811a8
    # https://stackoverflow.com/questions/35774060/determine-which-wtforms-button-was-pressed-in-a-flask-view
    # https://stackoverflow.com/questions/23283348/validate-wtform-form-based-on-clicked-button
