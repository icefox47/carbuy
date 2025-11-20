from flask import Flask, request, jsonify, render_template, url_for
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
import webbrowser
from threading import Timer
from dotenv import load_dotenv
import google.generativeai as genai
import json

# Load environment variables
load_dotenv()

# Configure Gemini AI
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # Use gemini-1.5-flash (current supported model)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    gemini_model = None
    print("WARNING: GEMINI_API_KEY not found. Using mock LLM instead.")

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
    raw_query = db.Column(db.Text)  # New field for AI query
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
            'raw_query': self.raw_query,
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

def gemini_llm_process(query):
    """
    Use Google Gemini AI to parse natural language query into structured data.
    Falls back to mock implementation if API key is not configured.
    """
    if not gemini_model:
        # Fallback to mock implementation
        return mock_llm_process(query)
    
    try:
        prompt = f"""You are a car search assistant. Extract car preferences from the user's query and return ONLY a JSON object with these fields:
- make: car manufacturer (e.g., "Toyota", "Honda", "BMW") or null
- type: body type (e.g., "suv", "sedan", "hatchback", "coupe", "truck", "van") or null
- year: year as string (e.g., "2024", "2023") or null
- budget: one of ["below_1lac", "1_2lac", "2_3lac", "3_5lac", "above_5lac"] or null (1 lac = 100,000 AED)
- condition: "new" or "used" or null
- fuel: fuel type ("petrol", "diesel", "electric", "cng") or null
- transmission: "automatic" or "manual" or null

User query: "{query}"

Return ONLY valid JSON, no other text. If a field cannot be determined, use null.

Example output:
{{"make": "Toyota", "type": "suv", "year": null, "budget": "below_1lac", "condition": "used", "fuel": null, "transmission": null}}"""

        response = gemini_model.generate_content(prompt)
        result_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if result_text.startswith('```'):
            result_text = result_text.split('```')[1]
            if result_text.startswith('json'):
                result_text = result_text[4:]
            result_text = result_text.strip()
        
        # Parse JSON response
        result = json.loads(result_text)
        
        # Ensure all expected fields exist
        default_result = {
            'make': None,
            'type': None,
            'year': None,
            'budget': None,
            'condition': None,
            'fuel': None,
            'transmission': None
        }
        default_result.update(result)
        
        return default_result
        
    except Exception as e:
        print(f"Gemini API Error: {e}")
        # Fallback to mock implementation on error
        return mock_llm_process(query)

def mock_llm_process(query):
    """
    Mock LLM function to parse natural language query into structured data.
    Used as fallback when Gemini API is not available.
    """
    query = query.lower()
    result = {
        'make': None,
        'type': None,
        'year': None,
        'budget': None,
        'condition': None,
        'fuel': None,
        'transmission': None
    }
    
    # Simple keyword matching for demo purposes
    makes = ['toyota', 'honda', 'maruti', 'bmw', 'hyundai', 'tata', 'ford']
    types = ['suv', 'sedan', 'hatchback', 'coupe', 'truck', 'van']
    conditions = ['new', 'used']
    fuels = ['petrol', 'diesel', 'electric', 'cng']
    
    for make in makes:
        if make in query:
            result['make'] = make.capitalize()
            break
            
    for car_type in types:
        if car_type in query:
            result['type'] = car_type
            break
            
    for condition in conditions:
        if condition in query:
            result['condition'] = condition
            break
            
    for fuel in fuels:
        if fuel in query:
            result['fuel'] = fuel
            break
            
    # Extract year (simple 4 digit search)
    import re
    year_match = re.search(r'\b20\d{2}\b', query)
    if year_match:
        result['year'] = year_match.group()
        
    # Extract budget (very basic)
    if '100k' in query or '1 lac' in query or '100000' in query:
        result['budget'] = 'below_1lac'
    elif '200k' in query or '2 lac' in query:
        result['budget'] = '1_2lac'
    # Add more budget logic as needed
    
    return result

@app.route('/api/ai-search', methods=['POST'])
def ai_search():
    try:
        data = request.get_json()
        query = data.get('query')
        user_name = data.get('name')
        user_phone = data.get('phone')
        
        if not query or not user_name or not user_phone:
            return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400
            
        # Process query with Gemini AI (or fallback to mock)
        structured_criteria = gemini_llm_process(query)
        
        # Create inquiry record
        inquiry = CarInquiry(
            name=user_name,
            phone=user_phone,
            make=structured_criteria.get('make', 'Not Specified'),
            type=structured_criteria.get('type', 'Not Specified'),
            year=structured_criteria.get('year', 'Not Specified'),
            condition=structured_criteria.get('condition', 'Not Specified'),
            kms_driven='Not Specified',
            fuel=structured_criteria.get('fuel', 'Not Specified'),
            transmission=structured_criteria.get('transmission', 'Not Specified'),
            budget=structured_criteria.get('budget', 'Not Specified'),
            raw_query=query
        )
        
        db.session.add(inquiry)
        db.session.commit()
        
        # Get recommendations based on structured criteria
        # Reuse the logic from submit_inquiry or create a shared helper
        # For now, let's just do a simple filter here similar to submit_inquiry
        
        # ... (Recommendation logic similar to submit_inquiry but using structured_criteria)
        # For brevity in this step, I will just return the structured criteria and let the frontend redirect or show results
        # But the user asked to "generate results". So let's fetch them.
        
        recommended_cars = []
        # Simple filter
        for car in SAMPLE_CARS:
            score = 0
            if structured_criteria['make'] and car['make'].lower() == structured_criteria['make'].lower():
                score += 3
            if structured_criteria['type'] and car['type'].lower() == structured_criteria['type'].lower():
                score += 2
            if structured_criteria['year'] and car['year'] == structured_criteria['year']:
                score += 1
            
            if score > 0:
                recommended_cars.append(car)
                
        # If no matches, return some defaults
        if not recommended_cars:
            recommended_cars = SAMPLE_CARS[:3]
            
        return jsonify({
            'status': 'success',
            'message': 'Search processed successfully',
            'data': inquiry.to_dict(),
            'criteria': structured_criteria,
            'recommendations': recommended_cars
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/submit-inquiry', methods=['POST'])
def submit_inquiry():
    try:
        data = request.get_json()
        
        # Handle skipped fields by providing defaults
        year = data.get('year') or 'Not Specified'
        kms_driven = data.get('kms_driven') or 'Not Specified'

        # Create new inquiry with fallback values
        inquiry = CarInquiry(
            name=data['name'],
            phone=data['phone'],
            make=data.get('make', 'Not Specified'),
            type=data.get('type', 'Not Specified'),
            year=year,
            condition=data.get('condition', 'Not Specified'),
            kms_driven=kms_driven,
            fuel=data.get('fuel', 'Not Specified'),
            transmission=data.get('transmission', 'Not Specified'),
            budget=data.get('budget', 'Not Specified'),
            raw_query=data.get('raw_query') # Support raw query if passed here too
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
