import React from 'react';
import { Link, NavLink } from 'react-router-dom';
import HomeOutlinedIcon from '@mui/icons-material/HomeOutlined';
import PersonOutlineOutlinedIcon from '@mui/icons-material/PersonOutlineOutlined';
import StyleOutlinedIcon from '@mui/icons-material/StyleOutlined';
import { USER, initials } from '../user';

const LINKS = [
  { to: '/home', label: 'Home', Icon: HomeOutlinedIcon },
  { to: '/profile', label: 'Profile', Icon: PersonOutlineOutlinedIcon },
  { to: '/', label: 'Swipe', Icon: StyleOutlinedIcon, end: true },
];

const Navbar = () => {
  return (
    <nav className="sidenav" aria-label="Main">
      <Link to="/" className="sidenav__brand display">
        FINDYOURFIT<sup>^*</sup>
      </Link>

      <ul className="sidenav__links">
        {LINKS.map(({ to, label, Icon, end }) => (
          <li key={to}>
            <NavLink to={to} end={end} className={({ isActive }) => `sidenav__link${isActive ? ' active' : ''}`}>
              <Icon fontSize="small" />
              {label}
            </NavLink>
          </li>
        ))}
      </ul>

      <div className="sidenav__user">
        <span className="avatar" aria-hidden="true">{initials(USER.name)}</span>
        <div>
          <div className="sidenav__name">{USER.name}</div>
          <div className="sidenav__sub">Personal stylist session</div>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
