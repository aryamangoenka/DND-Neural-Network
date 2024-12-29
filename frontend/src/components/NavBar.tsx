import React from "react";
import { NavLink } from "react-router-dom";
import "../styles/components/NavBar.scss";

const NavBar: React.FC = () => {
    return (
        <nav className="navbar">
            <div className="container d-flex justify-content-center">
                <ul className="navbar-nav d-flex flex-wrap flex-row justify-content-center">
                    {["Home", "Build", "Train", "Share"].map((label, index) => {
                        const route = label.toLowerCase() === "home" ? "/" : `/${label.toLowerCase()}`;
                        return (
                            <li className="nav-item mx-3" key={index}>
                                <NavLink
                                    to={route}
                                    className={({ isActive }) =>
                                        isActive ? "nav-link active" : "nav-link"
                                    }
                                >
                                    {label}
                                </NavLink>
                            </li>
                        );
                    })}
                </ul>
            </div>
        </nav>
    );
};

export default NavBar;
