import React from "react";
import { Link } from "react-router-dom";
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import '../Home.css';
import tshirts from '../assets/tshirts_card.jpeg';
import jackets from '../assets/jackets_card.jpeg';
import skirt from '../assets/skirt_card.jpeg';
import pants from '../assets/pants_card.jpeg';
import dress from '../assets/dress_card.jpeg';
import ath from '../assets/ath_card.jpeg';
import logo from '../assets/Logo1.png'

// Each category opens the swipe deck steered towards it (FashionCLIP text steering).
const CATEGORIES = [
  { label: 'T-Shirts', steer: 't-shirts', img: tshirts },
  { label: 'Jackets', steer: 'jackets', img: jackets },
  { label: 'Skirts', steer: 'skirts', img: skirt },
  { label: 'Pants', steer: 'trousers', img: pants },
  { label: 'Dresses', steer: 'dresses', img: dress },
  { label: 'Athleisure', steer: 'sportswear', img: ath },
];

const Home = () => {
  return (
    <div className="home">
      <section className="home__hero">
        <img src={logo} className="home__logo" alt="FindYourFit" />
        <div className="home__heroText">
          <p>Swipe through H&amp;M&apos;s current collection. Every like and pass teaches your personal recommender what you love.</p>
          <Link to="/" className="btn-primary-mono">Start swiping <ArrowForwardIcon fontSize="small" /></Link>
        </div>
      </section>

      <h2 className="home__title">Browse by category</h2>
      <div className="home__grid">
        {CATEGORIES.map(({ label, steer, img }) => (
          <Link key={label} to={`/?steer=${encodeURIComponent(steer)}`} className="home__card">
            <img src={img} alt="" />
            <span className="home__cardLabel">
              {label}
              <ArrowForwardIcon fontSize="small" />
            </span>
          </Link>
        ))}
      </div>
    </div>
  );
};

export default Home;
