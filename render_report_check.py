from main import app
from flask import render_template

# Data matching what generate_report produces
name = 'Test Patient'
dob = '1990-01-01'
age = 35
disease_name = 'Breast Cancer'
clinical_history = 'No major history.'
symptoms = ['none']
prepared_by = 'Tester'
image_paths = ['uploads/test_visualization_input_output.png']
diseases_level = ['Cancer: No (Non-Malignant)']
# Minimal data structure expected
data = {
    'clinical examination': 'N/A',
    'imaging studies': ['Mammography: Sample finding'],
    'pathological staging': 'N/A',
    'Recommended diet': ['A','B'],
    'Recommended exercise': ['A','B'],
    'precautions': ['A']
}
current_date = '2025-11-10'
user_type = 'patient'

with app.test_request_context('/'):
    html = render_template('report.html', user_type=user_type, name=name, dob=dob, age=age,
                           disease_name=disease_name, clinical_history=clinical_history,
                           symptoms=symptoms, prepared_by=prepared_by, image_paths=image_paths,
                           diseases_level=diseases_level, data=data, current_date=current_date)

    # Simple check for the img tag
    expected = '/static/' + image_paths[0]
    contains = expected in html
    print('Rendered HTML contains expected image path:', contains)
    if not contains:
        # Print snippet around where image path would be
        start = html.find('img')
        print(html[start:start+400])
    else:
        start = html.find(expected)
        snippet = html[max(0,start-80):start+80]
        print('Snippet:', snippet)
