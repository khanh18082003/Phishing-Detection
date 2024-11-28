const UrlForm = () => {
  return (
    <form action="" method="POST">
      <label htmlFor="url" className="block mb-2">
        Url Path:
      </label>
      <input
        name="url-path"
        id="url"
        placeholder="Enter url"
        className="block w-full h-10 rounded-md outline-none px-3"
      />
      <input
        type="submit"
        value={"Check"}
        className="block bg-[#0D47A1] text-white px-5 py-2 mx-auto text-[16px] rounded-lg my-4 tracking-[2px] cursor-pointer"
      />
    </form>
  );
};

export default UrlForm;
