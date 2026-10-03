import React from 'react'

import CloseIcon from '@mui/icons-material/Close';
import FavoriteIcon from '@mui/icons-material/Favorite';
import ShoppingBagOutlinedIcon from '@mui/icons-material/ShoppingBagOutlined';
import IconButton from '@mui/material/IconButton';
import '../SwipeButtons.css'

// The like/dislike buttons swipe the active TinderCard programmatically, so they
// go through exactly the same feedback -> recommendation path as a drag swipe.
const SwipeButtons = ({ onDislike, onLike, onCart, disabled }) => {
  return (
    <>
        <div className="swipeButtons">
            <IconButton className='swipeButtons__left' onClick={onDislike} disabled={disabled} aria-label="Dislike">
            <CloseIcon fontSize="large"/>
            </IconButton>
            <IconButton className='swipeButtons__cart' onClick={onCart} disabled={disabled} aria-label="Add to cart">
            <ShoppingBagOutlinedIcon fontSize="medium"/>
            </IconButton>
            <IconButton className='swipeButtons__right' onClick={onLike} disabled={disabled} aria-label="Like">
            <FavoriteIcon fontSize="large"/>
            </IconButton>


        </div>
    </>
  )
}

export default SwipeButtons
