import reactLogo from "../../assets/react.svg";
import MenuItem from "./MenuItem";
import { useState } from "react";

import items from "./menu-item.json";
import Form from "../PhishingEmail/Form";

const Header = () => {
  const [itemActive, setItemActive] = useState(0);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const handleClickedItem = (index) => {
    setItemActive(index);
    setIsModalOpen(true);
  };
  const closeModal = () => {
    setIsModalOpen(false); // Ẩn modal
  };

  return (
    <div>
      <div className="container">
        <div className="flex items-center justify-between">
          <div className="w-full max-w-1/6 px-[15px]">
            <div className="py-[25px]">
              <a href="#" className="inline-block">
                <img src={reactLogo} alt="react image" />
              </a>
            </div>
          </div>

          <div className="flex w-full max-w-5/6 px-[15px]">
            <div className="nav-menu">
              <ul className="main-menu">
                {items.items.map((item, index) => (
                  <MenuItem
                    key={index}
                    className={`${itemActive == index ? "active" : ""}`}
                    onClick={() => handleClickedItem(index)}
                  >
                    {item}
                  </MenuItem>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>
      {isModalOpen && (
        <Form
          nameForm={"Phishing Mail"}
          onClick={closeModal}
          isModalOpen={true}
          itemActive={itemActive}
        ></Form>
      )}
    </div>
  );
};

export default Header;
