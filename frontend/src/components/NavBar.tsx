import React from "react";
import { NavLink } from "react-router-dom";
import "../styles/components/NavBar.scss";

const NavBar: React.FC = () => {
  return (
    <nav className="navbar">
      <div className="container d-flex justify-content-between align-items-center">
        <div className="navbar-brand">
          <i className="fas fa-brain brand-icon"></i>
          <span className="brand-text">NeuroBlock</span>
        </div>

        <div className="navbar-actions">
          <NavLink
            to="/tutorial"
            className="tutorial-button"
            title="Learn how to use DND Neural Network"
          >
            <i className="fas fa-question-circle"></i>
            <span>Tutorial</span>
          </NavLink>
        </div>
      </div>
    </nav>
  );
};

export default NavBar;
