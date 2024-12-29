import React from "react";
import { NavLink } from "react-router-dom";

const NavBar: React.FC = () => {
    return (
        <nav className="navbar navbar-dark bg-dark">
            <div className="container d-flex justify-content-center">
                <ul className="navbar-nav d-flex flex-row">
                    <li className="nav-item mx-3">
                        <NavLink
                            to="/"
                            className={({ isActive }) =>
                                isActive ? "nav-link active" : "nav-link"
                            }
                        >
                            Home
                        </NavLink>
                    </li>
                    <li className="nav-item mx-3">
                        <NavLink
                            to="/build"
                            className={({ isActive }) =>
                                isActive ? "nav-link active" : "nav-link"
                            }
                        >
                            Build
                        </NavLink>
                    </li>
                    <li className="nav-item mx-3">
                        <NavLink
                            to="/train"
                            className={({ isActive }) =>
                                isActive ? "nav-link active" : "nav-link"
                            }
                        >
                            Train
                        </NavLink>
                    </li>
                    <li className="nav-item mx-3">
                        <NavLink
                            to="/export"
                            className={({ isActive }) =>
                                isActive ? "nav-link active" : "nav-link"
                            }
                        >
                            Share
                        </NavLink>
                    </li>
                </ul>
            </div>
        </nav>
    );
};

export default NavBar;
