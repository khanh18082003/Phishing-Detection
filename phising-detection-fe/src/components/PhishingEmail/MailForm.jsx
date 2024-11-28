const MailForm = (props) => {
  return (
    <form action="" method="POST" onSubmit={props.handleSubmitForm}>
      <label htmlFor="email" className="block mb-2">
        Email Content:
      </label>
      <textarea
        name="emailContent"
        id="email"
        placeholder="Enter body part of email"
        onChange={props.handleChange}
        className="block w-full h-[150px] rounded-md outline-none px-3 py-2"
      ></textarea>
      <input
        type="submit"
        value={"Check"}
        className="block bg-[#0D47A1] text-white px-5 py-2 mx-auto text-[16px] rounded-lg my-4 tracking-[2px] cursor-pointer"
      />
    </form>
  );
};

export default MailForm;
