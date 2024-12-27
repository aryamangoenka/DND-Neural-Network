import React from "react";
import { Link } from "react-router-dom";

const NavBar: React.FC = () => {
    return (
        <nav className="navbar navbar-expand-lg navbar-light bg-light mb-3">
            <div className="container">
                <Link className="nav-link" to="/">
                    homepage
                </Link>
                <Link className="nav-link" to="/build">
                    Build Neural Network
                </Link>
                <Link className="nav-link" to="/train">
                    Train
                </Link>
                <Link className="nav-link" to="/export">
                    Export
                </Link>
            </div>
        </nav>
    );
};

export default NavBar;
