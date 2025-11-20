from app import app, db, CarInquiry
import inspect

with app.app_context():
    db.create_all()
    inspector = db.inspect(db.engine)
    columns = [col['name'] for col in inspector.get_columns('car_inquiry')]
    print("Columns in CarInquiry table:")
    for col in columns:
        print(f"- {col}")
        
    expected_columns = ['id', 'name', 'phone', 'raw_query', 'enhanced_query', 'interested_cars', 'created_at']
    
    # Check if we have exactly these columns (ignoring order)
    if set(columns) == set(expected_columns):
        print("\nSUCCESS: Schema matches simplified requirements.")
    else:
        print(f"\nERROR: Schema mismatch.\nExpected: {expected_columns}\nFound: {columns}")
