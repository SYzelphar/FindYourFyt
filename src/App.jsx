import './App.css';
import Navbar from './Components/Navbar';
import {
  BrowserRouter as Router,
  Routes,
  Route,
  useLocation, // Keep the useLocation here
} from "react-router-dom";
import Swipe from './Components/Swipe';
import CartSidebar from './Components/CartSidebar';
import Home from './Components/Home';
import Profile from './Components/Profile';
import Footer from './Components/Footer';
import { CartProvider } from './CartContext';

function App() {
  return (
    <Router> {/* Move Router here */}
      <CartProvider>
        <LayoutWithFooterCheck />
      </CartProvider>
    </Router>
  );
}

// Create a new component that checks the location and renders the Footer conditionally
function LayoutWithFooterCheck() {
  const location = useLocation(); // Now this will be inside the Router

  return (
    <>
      <Navbar /> {/* Navbar will always be visible */}

      <Routes>
        <Route path="/home" element={<Home />} />
        <Route
          path="/"
          element={
            <>
              <Swipe /> {/* renders its own SwipeButtons, wired to the same swipe handler */}
              <CartSidebar />
            </>
          }
        />
        <Route path="/profile" element={<Profile />} />
      </Routes>

      {/* Conditionally render the Footer based on the route */}
      {location.pathname !== "/" && <Footer />} {/* No footer on the Swipe page */}
    </>
  );
}

export default App;