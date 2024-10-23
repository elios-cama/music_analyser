import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lyrics_analyzer.firebase_config import db

def test_firebase_connection():
    try:
        # Try to write to a test collection
        test_ref = db.collection('test').document('test_connection')
        test_ref.set({
            'message': 'Test connection successful',
            'test_value': 123
        })
        
        # Try to read it back
        doc = test_ref.get()
        if doc.exists:
            print("Successfully wrote and read from Firebase!")
            print("Retrieved data:", doc.to_dict())
            
            # Clean up by deleting the test document
            test_ref.delete()
            print("Test document cleaned up")
        else:
            print("Document does not exist!")
            
    except Exception as e:
        print(f"Error connecting to Firebase: {str(e)}")

if __name__ == "__main__":
    test_firebase_connection()