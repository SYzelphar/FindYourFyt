import React from 'react';
import ShoppingBagOutlinedIcon from '@mui/icons-material/ShoppingBagOutlined';
import '../CartSidebar.css';
import { useCart } from '../useCart';

const CartSidebar = () => {
  const { cart, removeFromCart } = useCart();

  return (
    <aside className="right-sidebar" aria-label="Cart">
      <div className="sidebar-title">
        <span>Your cart</span>
        {cart.length > 0 && <span className="sidebar-count">{cart.length}</span>}
      </div>
      {cart.length === 0 ? (
        <div className="cart-empty">
          <ShoppingBagOutlinedIcon />
          <p>Tap the bag button to save pieces you love.</p>
        </div>
      ) : (
        <ul className="menu">
          {cart.map((item) => (
            <li key={item.id} className="cart-item">
              <img src={item.image} alt={item.name} className="cart-item__img" />
              <span className="cart-item__name">{item.name}<small>{item.colour} · {item.productType}</small></span>
              <button className="cart-item__remove" onClick={() => removeFromCart(item.id)} aria-label={`Remove ${item.name}`}>×</button>
            </li>
          ))}
        </ul>
      )}
    </aside>
  );
};

export default CartSidebar;
