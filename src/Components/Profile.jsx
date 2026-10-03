import React from "react";
import '../Profile.css';
import StyleProfile from './StyleProfile';
import { USER, initials } from '../user';

const Profile = () => {
  return (
    <div className="profile-page">
      <div className="profile-container">
        <header className="profile-header">
          <div className="user-icon">{initials(USER.name)}</div>
          <div>
            <h2 className="profile-heading display">{USER.name}</h2>
            <p className="profile-sub">Your FindYourFit profile</p>
          </div>
        </header>

        <StyleProfile />

        <div className="profile-grid">
          <div className="profile-section">
            <h3 className="section-title">Address</h3>
            <div className="section-content">
              <p>Home: not set</p>
              <p>Office: not set</p>
              <button className="section-button">Manage addresses</button>
            </div>
          </div>

          <div className="profile-section">
            <h3 className="section-title">Payment methods</h3>
            <div className="section-content">
              <p>No payment method saved.</p>
              <button className="section-button">Add a payment method</button>
            </div>
          </div>

          <div className="profile-section">
            <h3 className="section-title">Past orders</h3>
            <div className="section-content">
              <p>Your past orders will appear here.</p>
              <button className="section-button">View all orders</button>
            </div>
          </div>

          <div className="profile-section">
            <h3 className="section-title">Account settings</h3>
            <div className="section-content">
              <p>Email: ************@example.com</p>
              <p>Phone: not set</p>
              <button className="section-button">Edit account details</button>
            </div>
          </div>

          <div className="profile-section">
            <h3 className="section-title">Your most chosen websites</h3>
            <div className="section-content">
              <p>Website 1</p>
              <p>Website 2</p>
              <button className="section-button">Manage websites</button>
            </div>
          </div>

          <div className="profile-section">
            <h3 className="section-title">Help &amp; support</h3>
            <div className="section-content">
              <p>Questions about your recommendations?</p>
              <button className="section-button">Get help</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Profile;
