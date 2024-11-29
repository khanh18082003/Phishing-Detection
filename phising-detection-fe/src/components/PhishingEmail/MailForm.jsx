const MailForm = (props) => {
  return (
    <form action="" method="POST" onSubmit={props.handleSubmitForm}>
      <label htmlFor="file" className="block mb-2 mt-4">
        Attach File (.eml):
      </label>
      <input
        type="file"
        id="file"
        name="file_name"
        onChange={props.handleChange}
        accept=".eml"
        required
      />
      <input
        type="submit"
        value={"Check"}
        className="block bg-[#0D47A1] text-white px-5 py-2 mx-auto text-[16px] rounded-lg my-4 tracking-[2px] cursor-pointer"
      />
    </form>
  );
};

export default MailForm;
