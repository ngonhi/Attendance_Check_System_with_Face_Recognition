$(function(){
	$('#btnSignUp').click(function(){
		$.ajax({
			url: '/register/',
			data: $('form').serialize(),
			type: 'POST',
			success: function(response){
				console.log(response);
				window.location.href = "http://localhost:5000/showGetFaces/";
			},
			error: function(error){
				console.log(error);
			}
		});
	});
});

$(function(){
	$('#btnFinish').click(function(){
		$.ajax({
			url: '/home',
			type: 'GET',
			success: function(response){
				console.log(response);
				window.location.href = "http://localhost:5000/home";
			},
			error: function(error){
				console.log(error);
			}
		});
	});
});
