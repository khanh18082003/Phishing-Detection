import { useState } from "react";
import MailForm from "./MailForm";
import UrlForm from "./UrlForm";
import axios from "axios";

const Form = (props) => {
  const [formData, setFormData] = useState({});
  const [result, setResult] = useState(0);
  const [visible, setVisible] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value,
    });
    console.log(formData);
  };
  const handleSubmitForm = (e) => {
    e.preventDefault();
    axios
      .post("http://127.0.0.1:8000/api/predict/", {
        input: formData, // Dữ liệu đầu vào
      })
      .then((response) => {
        console.log("Prediction result:", response.data.prediction);
        setResult(response.data.prediction);
        setVisible(true);
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  };
  return (
    <div className="modal fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 ">
      <div className="bg-slate-300 w-[500px] mx-auto rounded-xl px-9 py-4 relative">
        <button
          onClick={props.onClick}
          className="absolute top-2 right-2 text-gray-500 hover:text-gray-800 text-[30px]"
        >
          &times;
        </button>
        <h3 className="text-center text-[20px] font-semibold mb-4">
          {props.nameForm}
        </h3>
        {props.itemActive === 0 ? (
          <MailForm
            handleSubmitForm={handleSubmitForm}
            handleChange={handleChange}
          ></MailForm>
        ) : (
          <UrlForm
            handleSubmitForm={handleSubmitForm}
            handleChange={handleChange}
          ></UrlForm>
        )}

        <div className={`${visible ? "block" : "hidden"}`}>
          Result: <span>{result}</span>
        </div>
      </div>
    </div>
  );
};

export default Form;
