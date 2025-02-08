from flask import Flask, request, jsonify, render_template, url_for
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
import webbrowser
from threading import Timer

app = Flask(__name__)
CORS(app)

# Database Configuration
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'car_inquiries.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Models
class CarInquiry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    make = db.Column(db.String(50))
    type = db.Column(db.String(50))
    year = db.Column(db.String(4))
    condition = db.Column(db.String(20))
    kms_driven = db.Column(db.String(50))
    fuel = db.Column(db.String(20))
    transmission = db.Column(db.String(20))
    budget = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'phone': self.phone,
            'make': self.make,
            'type': self.type,
            'year': self.year,
            'condition': self.condition,
            'kms_driven': self.kms_driven,
            'fuel': self.fuel,
            'transmission': self.transmission,
            'budget': self.budget,
            'created_at': self.created_at.isoformat()
        }

# Sample car data (in a real app, this would come from a database)
SAMPLE_CARS = [
    {
        'id': 1,
        'make': 'Toyota',
        'model': 'Innova Crysta',
        'type': 'suv',
        'year': '2024',
        'condition': 'new',
        'price': 2500000,
        'fuel': 'diesel',
        'transmission': 'automatic',
        'kms_driven': 'below_25000',
        'image': 'https://images.unsplash.com/photo-1621007947382-bb3c3994e3fb?w=500',
        'features': ['360 Camera', 'Sunroof', 'Cruise Control', 'Ventilated Seats']
    },
    {
        'id': 2,
        'make': 'Honda',
        'model': 'City',
        'type': 'sedan',
        'year': '2023',
        'condition': 'used',
        'price': 1200000,
        'fuel': 'petrol',
        'transmission': 'manual',
        'kms_driven': '25000_50000',
        'image': 'https://images.unsplash.com/photo-1519641471654-76ce0107ad1b?w=500',
        'features': ['Push Start', 'Sunroof', 'Apple CarPlay', 'Rear AC']
    },
    {
        'id': 3,
        'make': 'Maruti Suzuki',
        'model': 'Swift',
        'type': 'sedan',
        'year': '2024',
        'condition': 'new',
        'price': 800000,
        'fuel': 'cng',
        'transmission': 'manual',
        'kms_driven': 'below_25000',
        'image': 'https://images.unsplash.com/photo-1583121274602-3e2820c69888?w=500',
        'features': ['Touchscreen', 'Rear Camera', 'Alloy Wheels']
    },
    {
        'id': 4,
        'make': 'BMW',
        'model': '3 Series',
        'type': 'sedan',
        'year': '2023',
        'condition': 'used',
        'price': 4500000,
        'fuel': 'petrol',
        'transmission': 'automatic',
        'kms_driven': '50000_75000',
        'image': 'https://images.unsplash.com/photo-1523983388277-336a66bf9bcd?w=500',
        'features': ['Premium Sound', 'Leather Seats', 'Navigation', 'Sunroof']
    },
    {
        'id': 5,
        'make': 'Hyundai',
        'model': 'Creta',
        'type': 'suv',
        'year': '2024',
        'condition': 'new',
        'price': 1800000,
        'fuel': 'diesel',
        'transmission': 'automatic',
        'kms_driven': 'below_25000',
        'image': 'https://images.unsplash.com/photo-1606664515524-ed2f786a0bd6?w=500',
        'features': ['Panoramic Sunroof', 'Ventilated Seats', 'ADAS', 'Digital Cluster']
    },
    {
        'id': 6,
        'make': 'Tata',
        'model': 'Nexon EV',
        'type': 'suv',
        'year': '2024',
        'condition': 'new',
        'price': 1500000,
        'fuel': 'electric',
        'transmission': 'automatic',
        'kms_driven': 'below_25000',
        'image': 'https://images.unsplash.com/photo-1606664515524-ed2f786a0bd6?w=500',
        'features': ['Electric Sunroof', 'Connected Car', 'Fast Charging', 'Air Purifier']
    },
    {
        'id': 7,
        'make': 'Ford',
        'model': 'Mustang',
        'type': 'coupe',
        'year': '2023',
        'condition': 'used',
        'price': 7500000,
        'fuel': 'petrol',
        'transmission': 'automatic',
        'kms_driven': '25000_50000',
        'image': 'https://images.unsplash.com/photo-1606664515524-ed2f786a0bd6?w=500',
        'features': ['V8 Engine', 'Track Mode', 'Premium Sound', 'Launch Control']
    }
]

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/car-form')
def car_form():
    # Get query parameters
    make = request.args.get('make', '')
    car_type = request.args.get('type', '')
    year = request.args.get('year', '')
    
    return render_template('car-form.html', make=make, type=car_type, year=year)

@app.route('/api/submit-inquiry', methods=['POST'])
def submit_inquiry():
    try:
        data = request.get_json()
        
        # Create new inquiry
        inquiry = CarInquiry(
            name=data['name'],
            phone=data['phone'],
            make=data['make'],
            type=data['type'],
            year=data['year'],
            condition=data['condition'],
            kms_driven=data['kms_driven'],
            fuel=data['fuel'],
            transmission=data['transmission'],
            budget=data['budget']
        )
        
        # Save to database
        db.session.add(inquiry)
        db.session.commit()

        # Convert budget range to actual price range
        budget_ranges = {
            'below_1lac': (0, 100000),
            '1_2lac': (100000, 200000),
            '2_3lac': (200000, 300000),
            '3_5lac': (300000, 500000),
            'above_5lac': (500000, float('inf'))
        }
        
        selected_budget_range = budget_ranges.get(data['budget'], (0, float('inf')))
        
        # Filter cars based on user preferences with more flexible matching
        recommended_cars = []
        
        # First try exact matches
        exact_matches = [
            car for car in SAMPLE_CARS
            if (not data['make'] or car['make'].lower() == data['make'].lower()) and
               (not data['type'] or car['type'].lower() == data['type'].lower()) and
               (not data['year'] or car['year'] == data['year']) and
               (not data['condition'] or car['condition'].lower() == data['condition'].lower()) and
               (not data['fuel'] or car['fuel'].lower() == data['fuel'].lower()) and
               (not data['transmission'] or car['transmission'].lower() == data['transmission'].lower()) and
               (not data['kms_driven'] or car['kms_driven'] == data['kms_driven']) and
               selected_budget_range[0] <= car['price'] <= selected_budget_range[1]
        ]
        
        recommended_cars.extend(exact_matches)
        
        # If we have less than 3 exact matches, add similar cars
        if len(recommended_cars) < 3:
            # Add cars of same make but different type/year
            if data['make']:
                make_matches = [
                    car for car in SAMPLE_CARS
                    if car['make'].lower() == data['make'].lower() and
                    car not in recommended_cars and
                    selected_budget_range[0] <= car['price'] <= selected_budget_range[1]
                ]
                recommended_cars.extend(make_matches[:2])
            
            # Add cars of same type but different make
            if len(recommended_cars) < 3 and data['type']:
                type_matches = [
                    car for car in SAMPLE_CARS
                    if car['type'].lower() == data['type'].lower() and
                    car not in recommended_cars and
                    selected_budget_range[0] <= car['price'] <= selected_budget_range[1]
                ]
                recommended_cars.extend(type_matches[:2])
            
            # Add cars in same budget range
            if len(recommended_cars) < 3:
                budget_matches = [
                    car for car in SAMPLE_CARS
                    if selected_budget_range[0] <= car['price'] <= selected_budget_range[1] and
                    car not in recommended_cars
                ]
                recommended_cars.extend(budget_matches[:2])
        
        # If we still have no recommendations, return cars in similar price range
        if not recommended_cars:
            recommended_cars = sorted(
                SAMPLE_CARS,
                key=lambda x: abs(x['price'] - selected_budget_range[1])
            )[:3]
        
        return jsonify({
            'status': 'success',
            'message': 'Inquiry submitted successfully',
            'data': inquiry.to_dict(),
            'recommendations': recommended_cars
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/api/inquiries', methods=['GET'])
def get_inquiries():
    inquiries = CarInquiry.query.order_by(CarInquiry.created_at.desc()).all()
    return jsonify([inquiry.to_dict() for inquiry in inquiries])

def init_db():
    with app.app_context():
        db.create_all()


if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)
