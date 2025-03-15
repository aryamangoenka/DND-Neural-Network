import React from "react";
import { NavLink } from "react-router-dom";
import "../styles/components/NavBar.scss";

const NavBar: React.FC = () => {
  return (
    <nav className="navbar">
      <div className="container d-flex justify-content-between align-items-center">
        <div className="navbar-brand">
          <span className="brand-text">DND Neural Network</span>
        </div>


      </div>
    </nav>
  );
};

export default NavBar;
