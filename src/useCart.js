import { createContext, useContext } from 'react';

export const CartContext = createContext({ cart: [], addToCart: () => {}, removeFromCart: () => {} });

export const useCart = () => useContext(CartContext);
