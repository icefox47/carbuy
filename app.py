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
    # Use gemini-pro (stable model)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
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
    raw_query = db.Column(db.Text)  # Original Prompt
    enhanced_query = db.Column(db.Text)  # Rewritten/Corrected Prompt
    interested_cars = db.Column(db.Text, default='')  # Comma-separated list of "Make Model"
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'phone': self.phone,
            'raw_query': self.raw_query,
            'enhanced_query': self.enhanced_query,
            'interested_cars': self.interested_cars,
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
        prompt = f"""You are a car search assistant.
1. Correct any spelling mistakes in the user's query and rewrite it clearly.
2. Extract car preferences from the user's query.

Return ONLY a JSON object with these fields:
- enhanced_query: The corrected and rewritten clear query string.
- make: car manufacturer or null
- type: body type or null
- year: year as string or null
- budget: one of ["below_1lac", "1_2lac", "2_3lac", "3_5lac", "above_5lac"] or null
- condition: "new" or "used" or null
- fuel: fuel type or null
- transmission: "automatic" or "manual" or null

User query: "{query}"

Return ONLY valid JSON, no other text.

Example output:
{{"enhanced_query": "I am looking for a used Toyota SUV under 100k", "make": "Toyota", "type": "suv", "year": null, "budget": "below_1lac", "condition": "used", "fuel": null, "transmission": null}}"""

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
            'enhanced_query': query, # Fallback to original
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
        'enhanced_query': query,
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

def generate_car_descriptions(query, cars, criteria):
    """
    Use Gemini AI to generate personalized descriptions for recommended cars.
    Explains why each car matches the user's needs.
    """
    if not gemini_model or not cars:
        return {}
    
    try:
        car_list = "\n".join([
            f"{i+1}. {car['year']} {car['make']} {car['model']} - AED {car['price']:,} ({car['condition']}, {car['fuel']}, {car['transmission']})"
            for i, car in enumerate(cars)
        ])
        
        prompt = f"""You are a helpful car sales assistant. A customer searched for: "{query}"

We understood they want: {criteria}

Here are the recommended cars:
{car_list}

For EACH car, write a brief (2-3 sentences) personalized explanation of why it matches their needs. Be specific about how it fits their criteria. Format as JSON:

{{
  "0": "explanation for first car",
  "1": "explanation for second car",
  ...
}}

Keep it friendly, helpful, and focused on matching their specific requirements."""

        response = gemini_model.generate_content(prompt)
        result_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if result_text.startswith('```'):
            result_text = result_text.split('```')[1]
            if result_text.startswith('json'):
                result_text = result_text[4:]
            result_text = result_text.strip()
        
        descriptions = json.loads(result_text)
        return descriptions
        
    except Exception as e:
        print(f"Error generating descriptions: {e}")
        return {}

def gemini_suggest_cars(query, criteria):
    """
    Use Gemini AI to intelligently suggest car models based on user query.
    Leverages Gemini's real-world knowledge of the car market.
    Handles misspellings and understands context naturally.
    """
    if not gemini_model:
        print("Gemini model not available, skipping AI suggestions")
        return []
    
    try:
        # Build criteria summary
        criteria_parts = []
        if criteria.get('type'): criteria_parts.append(f"type: {criteria['type']}")
        if criteria.get('make'): criteria_parts.append(f"make: {criteria['make']}")
        if criteria.get('condition'): criteria_parts.append(f"condition: {criteria['condition']}")
        if criteria.get('budget'): criteria_parts.append(f"budget: {criteria['budget']}")
        if criteria.get('fuel'): criteria_parts.append(f"fuel: {criteria['fuel']}")
        if criteria.get('transmission'): criteria_parts.append(f"transmission: {criteria['transmission']}")
        
        criteria_text = ", ".join(criteria_parts) if criteria_parts else "general preferences"
        
        prompt = f"""You are a car expert in the UAE/Middle East market. A customer is searching for a car.

User's query: "{query}"
Extracted criteria: {criteria_text}

Based on this query, suggest 5-7 specific car models that would be perfect matches. Consider:
- Popular models in UAE market
- Price ranges (1 lac = 100,000 AED)
- Family needs, business use, or other context from query
- Fuel efficiency, safety, reliability
- Handle any misspellings in the query naturally

For each suggestion, provide ONLY the following fields in JSON format:
- make: Car manufacturer (e.g., "Toyota", "Honda")
- model: Car model name (e.g., "Fortuner", "Civic")
- year: Year as string (e.g., "2024", "2023")
- price: Price in AED as number (e.g., 180000)
- condition: "new" or "used"
- fuel: "petrol", "diesel", "electric", or "cng"
- transmission: "automatic" or "manual"
- reason: Brief explanation why it matches (1-2 sentences)

Return ONLY a valid JSON array with NO additional text:
[
  {{
    "make": "Toyota",
    "model": "Fortuner",
    "year": "2024",
    "price": 180000,
    "condition": "new",
    "fuel": "diesel",
    "transmission": "automatic",
    "reason": "Perfect family SUV with excellent safety features and spacious interior"
  }}
]

Focus on realistic, available cars in the UAE market. Return ONLY the JSON array."""

        print(f"Calling Gemini for car suggestions...")
        response = gemini_model.generate_content(prompt)
        result_text = response.text.strip()
        print(f"Gemini response received: {result_text[:200]}...")
        
        # Remove markdown code blocks if present
        if result_text.startswith('```'):
            result_text = result_text.split('```')[1]
            if result_text.startswith('json'):
                result_text = result_text[4:]
            result_text = result_text.strip()
        
        suggestions = json.loads(result_text)
        
        # Add default images and features to each suggestion
        for i, car in enumerate(suggestions):
            # Add a generic car image (you can replace with real car images later)
            car['image'] = f"https://images.unsplash.com/photo-1{550+i}?w=500&auto=format"
            car['features'] = ['AI Recommended', 'Best Match', 'Popular Choice']
            car['kms_driven'] = 'below_25000' if car.get('condition') == 'new' else '25000_50000'
            car['id'] = i + 100  # Unique ID
        
        print(f"Successfully generated {len(suggestions)} AI suggestions")
        return suggestions
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error in car suggestions: {e}")
        print(f"Response text: {result_text if 'result_text' in locals() else 'N/A'}")
        return []
    except Exception as e:
        print(f"Error generating car suggestions: {e}")
        import traceback
        traceback.print_exc()
        return []

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
        
        # Use Gemini AI to intelligently suggest cars based on query
        ai_suggested_cars = gemini_suggest_cars(query, structured_criteria)
        
        if ai_suggested_cars:
            recommended_cars = ai_suggested_cars
        else:
            # Fallback logic (simplified for brevity, assuming SAMPLE_CARS exists)
            recommended_cars = SAMPLE_CARS[:3]

        # Create inquiry record with simplified fields
        inquiry = CarInquiry(
            name=user_name,
            phone=user_phone,
            raw_query=query,
            enhanced_query=structured_criteria.get('enhanced_query', query),
            interested_cars='' # Initialized empty
        )
        
        db.session.add(inquiry)
        db.session.commit()
        
        # Generate AI-powered personalized descriptions for each car
        car_descriptions = generate_car_descriptions(query, recommended_cars, structured_criteria)
            
        return jsonify({
            'status': 'success',
            'message': 'Search processed successfully',
            'inquiry_id': inquiry.id,
            'user_info': {
                'name': user_name,
                'phone': user_phone,
                'original_query': query
            },
            'data': inquiry.to_dict(),
            'criteria': structured_criteria,
            'recommendations': recommended_cars,
            'descriptions': car_descriptions
        })

    except Exception as e:
        db.session.rollback()
        print(f"Error in ai_search: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/log-interest', methods=['POST'])
def log_interest():
    try:
        data = request.get_json()
        inquiry_id = data.get('inquiry_id')
        car_details = data.get('car_details')
        
        if not inquiry_id or not car_details:
            return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400
            
        inquiry = CarInquiry.query.get(inquiry_id)
        if not inquiry:
            return jsonify({'status': 'error', 'message': 'Inquiry not found'}), 404
            
        # Update interested cars list (comma separated "Make Model")
        new_interest = f"{car_details.get('make')} {car_details.get('model')}"
        
        if inquiry.interested_cars:
            # Check if already present to avoid duplicates
            current_list = [item.strip() for item in inquiry.interested_cars.split(',')]
            if new_interest not in current_list:
                inquiry.interested_cars += f", {new_interest}"
        else:
            inquiry.interested_cars = new_interest
            
        db.session.commit()
            
        return jsonify({'status': 'success', 'message': 'Interest logged successfully'})
        
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
