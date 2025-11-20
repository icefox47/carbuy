from app import app, db, CarInquiry
import inspect

with app.app_context():
    db.create_all()
    inspector = db.inspect(db.engine)
    columns = [col['name'] for col in inspector.get_columns('car_inquiry')]
    print("Columns in CarInquiry table:")
    for col in columns:
        print(f"- {col}")
        
    expected_columns = ['criteria_json', 'recommendations_json', 'interested_cars_json', 'status']
    missing = [col for col in expected_columns if col not in columns]
    
    if missing:
        print(f"\nERROR: Missing columns: {missing}")
    else:
        print("\nSUCCESS: All new columns are present.")
