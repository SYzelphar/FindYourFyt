import React, { useState } from 'react';
import { CartContext } from './useCart';

export const CartProvider = ({ children }) => {
    const [cart, setCart] = useState([]);

    const addToCart = (product) =>
        setCart((items) => (items.some((p) => p.id === product.id) ? items : [...items, product]));

    const removeFromCart = (id) => setCart((items) => items.filter((p) => p.id !== id));

    return (
        <CartContext.Provider value={{ cart, addToCart, removeFromCart }}>
            {children}
        </CartContext.Provider>
    );
};
