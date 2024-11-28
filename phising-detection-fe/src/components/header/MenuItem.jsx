const MenuItem = (props) => {
  return (
    <li
      className={`menu-item ${props.className ? "active" : ""}`}
      onClick={props.onClick}
    >
      <a href="#">
        {props.children}
        <div className="menu-item-line"></div>
      </a>
    </li>
  );
};

export default MenuItem;
