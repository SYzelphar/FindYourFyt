// Import necessary modules from Firebase SDK
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import { getFirestore } from "firebase/firestore"; // Import Firestore module

// Your web app's Firebase configuration
const firebaseConfig = {
    apiKey: "AIzaSyCkQVWyl9jv0n9F1zOdpe8AqL3XlvbTgWc",
    authDomain: "findyourfit-ce407.firebaseapp.com",
    projectId: "findyourfit-ce407",
    storageBucket: "findyourfit-ce407.appspot.com",
    messagingSenderId: "995141123076",
    appId: "1:995141123076:web:6c14c734e2b24abfdf0951",
    measurementId: "G-9SWDWSC70P"
};

// Initialize Firebase
const firebaseApp = initializeApp(firebaseConfig);

// Initialize Firestore and Analytics
const database = getFirestore(firebaseApp);
getAnalytics(firebaseApp);

// Export the Firestore instance for use in other parts of your app
export default database;