import Header from "./components/header/Header";

const App = () => {
  document.title = "Phising | Home";
  return (
    <div>
      <Header></Header>
      <h1 className="text-center text-[24px] font-extrabold tracking-[2px]">
        PHISHING DETECTION
      </h1>
      <div className="bg-img"></div>
    </div>
  );
};

export default App;
