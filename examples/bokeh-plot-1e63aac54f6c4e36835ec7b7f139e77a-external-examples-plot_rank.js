(function() {
  var fn = function() {
    
    (function(root) {
      function now() {
        return new Date();
      }
    
      var force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
      
      
    
      var element = document.getElementById("bad0ed37-dc8c-4863-ad34-691786e3952a");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'bad0ed37-dc8c-4863-ad34-691786e3952a' but no matching script tag was found.")
        }
      
    
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error() {
          console.error("failed to load " + url);
        }
    
        for (var i = 0; i < css_urls.length; i++) {
          var url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error;
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js": "T2yuo9Oe71Cz/I4X9Ac5+gpEa5a8PpJCDlqKYO0CfAuEszu1JrXLl8YugMqYe3sM", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js": "98GDGJ0kOMCUMUePhksaQ/GYgB3+NH9h996V88sh3aOiUNX3N+fLXAtry6xctSZ6", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.3.min.js": "89bArO+nlbP3sgakeHjCo1JYxYR5wufVgA3IbUvDY+K7w4zyxJqssu7wVnfeKCq8"};
    
        for (var i = 0; i < js_urls.length; i++) {
          var url = js_urls[i];
          var element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error;
          element.async = false;
          element.src = url;
          if (url in hashes) {
            element.crossOrigin = "anonymous";
            element.integrity = "sha384-" + hashes[url];
          }
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.3.min.js"];
      var css_urls = [];
      
    
      var inline_js = [
        function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        
        function(Bokeh) {
          (function() {
            var fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                    
                  var docs_json = '{"164fe0d4-53e5-4a94-a7d0-307f478de958":{"roots":{"references":[{"attributes":{"bottom":{"value":3},"fill_color":{"value":"#d62728"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26696","type":"VBar"},{"attributes":{"source":{"id":"26677"}},"id":"26681","type":"CDSView"},{"attributes":{"ticks":[0,1,2,3]},"id":"26701","type":"FixedTicker"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26706","type":"VBar"},{"attributes":{"line_dash":[6],"location":0.48076923076923067},"id":"26710","type":"Span"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26678","type":"VBar"},{"attributes":{"below":[{"id":"26616"}],"center":[{"id":"26619"},{"id":"26623"},{"id":"26682"},{"id":"26688"},{"id":"26694"},{"id":"26700"}],"left":[{"id":"26620"}],"output_backend":"webgl","plot_height":331,"plot_width":441,"renderers":[{"id":"26680"},{"id":"26686"},{"id":"26692"},{"id":"26698"}],"title":{"id":"26703"},"toolbar":{"id":"26634"},"toolbar_location":null,"x_range":{"id":"26608"},"x_scale":{"id":"26612"},"y_range":{"id":"26610"},"y_scale":{"id":"26614"}},"id":"26607","subtype":"Figure","type":"Plot"},{"attributes":{"callback":null},"id":"26631","type":"HoverTool"},{"attributes":{"line_dash":[6],"location":2.4166666666666665},"id":"26694","type":"Span"},{"attributes":{"overlay":{"id":"26632"}},"id":"26626","type":"BoxZoomTool"},{"attributes":{},"id":"26751","type":"BasicTickFormatter"},{"attributes":{"bottom":{"value":2},"fill_alpha":{"value":0.1},"fill_color":{"value":"#2ca02c"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26691","type":"VBar"},{"attributes":{},"id":"26646","type":"LinearScale"},{"attributes":{"source":{"id":"26689"}},"id":"26693","type":"CDSView"},{"attributes":{"data":{"top":{"__ndarray__":"AAAAAAAAAECkcD0K1yMBQCa/WPKLpQFA6LSBThtoAkBqA5020OkCQCz5xZJfrANA8O7u7u5uBEDrUbgehWsDQOtRuB6FawNAC9ejcD0KA0BKfrHkF0sDQA102kCnDQRAqqqqqqoqA0BQG+i0gU4EQC+W/GLJrwRAThvotIFOBECuR+F6FC4EQE4b6LSBTgRAcD0K16PwBEA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26743"},"selection_policy":{"id":"26744"}},"id":"26689","type":"ColumnDataSource"},{"attributes":{"bottom":{"value":2},"fill_color":{"value":"#2ca02c"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26690","type":"VBar"},{"attributes":{"line_dash":[6],"location":0.41666666666666663},"id":"26682","type":"Span"},{"attributes":{"data_source":{"id":"26689"},"glyph":{"id":"26690"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26691"},"selection_glyph":null,"view":{"id":"26693"}},"id":"26692","type":"GlyphRenderer"},{"attributes":{},"id":"26658","type":"ResetTool"},{"attributes":{"line_dash":[6],"location":2.480769230769231},"id":"26722","type":"Span"},{"attributes":{"data":{"top":{"__ndarray__":"AAAAAAAA8D8OdNpApw30PxSuR+F6FPY/1AY6baDT9T8c6LSBThv4PxdLfrHkF/c/1QY6baDT9T+V/GLJL5b2P1jyiyW/WPc/43oUrkfh+T8ehetRuB75PxdLfrHkF/c/mJmZmZmZ9z8YrkfhehT2P1RVVVVVVfY/lfxiyS+W9j/gehSuR+H5P5iZmZmZmfc/kl8s+cWS9T8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26741"},"selection_policy":{"id":"26742"}},"id":"26683","type":"ColumnDataSource"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"26624"},{"id":"26625"},{"id":"26626"},{"id":"26627"},{"id":"26628"},{"id":"26629"},{"id":"26630"},{"id":"26631"}]},"id":"26634","type":"Toolbar"},{"attributes":{"bottom":{"value":1},"fill_color":{"value":"#ff7f0e"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26684","type":"VBar"},{"attributes":{},"id":"26736","type":"BasicTickFormatter"},{"attributes":{"data":{"top":{"__ndarray__":"EPzAD/zACUAg+IEf+IELQMEP/MAPfApAdmIndmKnC0A4ciM3ciMOQIZe6IVeaA1Ah17ohV5oDUDYiZ3YiR0NQD7wAz/wAw9Ae6EXeqEXDEAbuZEbuRELQHZiJ3ZipwtAeqEXeqEXDEB0IzdyIzcLQBu5kRu5EQtAFDuxEzsxCkByIzdyIzcLQBu5kRu5EQtAxU7sxE7sCkA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26760"},"selection_policy":{"id":"26761"}},"id":"26723","type":"ColumnDataSource"},{"attributes":{"bottom":{"value":1},"fill_alpha":{"value":0.1},"fill_color":{"value":"#ff7f0e"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26685","type":"VBar"},{"attributes":{"bottom":{"value":3},"fill_color":{"value":"#d62728"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26724","type":"VBar"},{"attributes":{"data_source":{"id":"26683"},"glyph":{"id":"26684"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26685"},"selection_glyph":null,"view":{"id":"26687"}},"id":"26686","type":"GlyphRenderer"},{"attributes":{},"id":"26734","type":"BasicTickFormatter"},{"attributes":{},"id":"26745","type":"Selection"},{"attributes":{},"id":"26617","type":"BasicTicker"},{"attributes":{"data":{"top":{"__ndarray__":"4Qd+4Af+BUDVSq3USq0EQBh6oRd6oQJAGHqhF3qhAkAUO7ETOzECQCu1Uiu10gRAd2IndmKnA0DFTuzETuwCQHIjN3IjNwNAJDdyIzfyA0Bu5EZu5MYCQB/4gR/4gQNAxU7sxE7sAkDTC73QCz0EQNALvdALPQRA0Au90As9BEB6oRd6oRcEQIIf+IEf+ARAhl7ohV5oBUA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26758"},"selection_policy":{"id":"26759"}},"id":"26717","type":"ColumnDataSource"},{"attributes":{},"id":"26608","type":"DataRange1d"},{"attributes":{"data_source":{"id":"26717"},"glyph":{"id":"26718"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26719"},"selection_glyph":null,"view":{"id":"26721"}},"id":"26720","type":"GlyphRenderer"},{"attributes":{"source":{"id":"26683"}},"id":"26687","type":"CDSView"},{"attributes":{"bottom":{"value":2},"fill_alpha":{"value":0.1},"fill_color":{"value":"#2ca02c"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26719","type":"VBar"},{"attributes":{"line_dash":[6],"location":1.4166666666666665},"id":"26688","type":"Span"},{"attributes":{"source":{"id":"26717"}},"id":"26721","type":"CDSView"},{"attributes":{"bottom":{"value":2},"fill_color":{"value":"#2ca02c"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26718","type":"VBar"},{"attributes":{},"id":"26663","type":"UndoTool"},{"attributes":{"data_source":{"id":"26677"},"glyph":{"id":"26678"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26679"},"selection_glyph":null,"view":{"id":"26681"}},"id":"26680","type":"GlyphRenderer"},{"attributes":{"line_dash":[6],"location":1.4807692307692308},"id":"26716","type":"Span"},{"attributes":{"children":[{"id":"26765"},{"id":"26763"}]},"id":"26766","type":"Column"},{"attributes":{"overlay":{"id":"26667"}},"id":"26662","type":"LassoSelectTool"},{"attributes":{},"id":"26661","type":"WheelZoomTool"},{"attributes":{"source":{"id":"26695"}},"id":"26699","type":"CDSView"},{"attributes":{},"id":"26664","type":"SaveTool"},{"attributes":{"ticks":[0,1,2,3]},"id":"26729","type":"FixedTicker"},{"attributes":{},"id":"26659","type":"PanTool"},{"attributes":{"axis_label":"Chain","formatter":{"id":"26734"},"ticker":{"id":"26701"}},"id":"26620","type":"LinearAxis"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"26658"},{"id":"26659"},{"id":"26660"},{"id":"26661"},{"id":"26662"},{"id":"26663"},{"id":"26664"},{"id":"26665"}]},"id":"26668","type":"Toolbar"},{"attributes":{},"id":"26757","type":"UnionRenderers"},{"attributes":{"overlay":{"id":"26666"}},"id":"26660","type":"BoxZoomTool"},{"attributes":{"bottom":{"value":3},"fill_alpha":{"value":0.1},"fill_color":{"value":"#d62728"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26725","type":"VBar"},{"attributes":{"axis":{"id":"26654"},"dimension":1,"ticker":null},"id":"26657","type":"Grid"},{"attributes":{"text":"mu"},"id":"26731","type":"Title"},{"attributes":{"data":{"top":{"__ndarray__":"6YVe6IVe4D9nZmZmZmbeP2dmZmZmZu4/WWqlVmql7D/eyI3cyI3YP7vQC73QC9U/uBM7sRM73T+vEzuxEzvdPyZ2Yid2Ytc/lxu5kRu52T8ZuZEbuZHfP5AbuZEbudk/QS/0Qi/04D8LwQ/8wA/cP5AbuZEbudk/q9RKrdRK4z9BL/RCL/TgPyZ2Yid2Ytc/USu1Uiu1wj8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26754"},"selection_policy":{"id":"26755"}},"id":"26705","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"26695"},"glyph":{"id":"26696"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26697"},"selection_glyph":null,"view":{"id":"26699"}},"id":"26698","type":"GlyphRenderer"},{"attributes":{},"id":"26625","type":"PanTool"},{"attributes":{},"id":"26651","type":"BasicTicker"},{"attributes":{},"id":"26743","type":"Selection"},{"attributes":{"axis":{"id":"26620"},"dimension":1,"ticker":null},"id":"26623","type":"Grid"},{"attributes":{"axis_label":"Rank (all chains)","formatter":{"id":"26751"},"major_label_overrides":{"0":"0","1":"1","2":"2","3":"3"},"ticker":{"id":"26651"}},"id":"26650","type":"LinearAxis"},{"attributes":{},"id":"26759","type":"UnionRenderers"},{"attributes":{"callback":null},"id":"26665","type":"HoverTool"},{"attributes":{"axis":{"id":"26650"},"ticker":null},"id":"26653","type":"Grid"},{"attributes":{"data":{"top":{"__ndarray__":"ZWZmZmZm7j9OG+i0gU7XP2cDnTbQad8/WfKLJb9Y2j9Bpw102kDTP17JL5b8Yt0/PW2g0wY60T9U8oslv1jaP1ws+cWSX9w/WlVVVVVV2T9SVVVVVVXZPzTQaQOdNuA/ZgOdNtBp3z9m8oslv1jaP0h+seQXS9Y/SH6x5BdL1j84baDTBjrRPz+nDXTaQNM/SH6x5BdL1j8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26739"},"selection_policy":{"id":"26740"}},"id":"26677","type":"ColumnDataSource"},{"attributes":{"bottom":{"value":3},"fill_alpha":{"value":0.1},"fill_color":{"value":"#d62728"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26697","type":"VBar"},{"attributes":{},"id":"26624","type":"ResetTool"},{"attributes":{"line_dash":[6],"location":3.4166666666666665},"id":"26700","type":"Span"},{"attributes":{"source":{"id":"26711"}},"id":"26715","type":"CDSView"},{"attributes":{},"id":"26742","type":"UnionRenderers"},{"attributes":{},"id":"26741","type":"Selection"},{"attributes":{},"id":"26614","type":"LinearScale"},{"attributes":{"axis_label":"Chain","formatter":{"id":"26749"},"ticker":{"id":"26729"}},"id":"26654","type":"LinearAxis"},{"attributes":{},"id":"26610","type":"DataRange1d"},{"attributes":{},"id":"26630","type":"SaveTool"},{"attributes":{},"id":"26758","type":"Selection"},{"attributes":{},"id":"26744","type":"UnionRenderers"},{"attributes":{"children":[[{"id":"26607"},0,0],[{"id":"26643"},0,1]]},"id":"26763","type":"GridBox"},{"attributes":{"axis_label":"Rank (all chains)","formatter":{"id":"26736"},"major_label_overrides":{"0":"0","1":"1","2":"2","3":"3"},"ticker":{"id":"26617"}},"id":"26616","type":"LinearAxis"},{"attributes":{},"id":"26648","type":"LinearScale"},{"attributes":{},"id":"26761","type":"UnionRenderers"},{"attributes":{},"id":"26756","type":"Selection"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26679","type":"VBar"},{"attributes":{},"id":"26754","type":"Selection"},{"attributes":{},"id":"26740","type":"UnionRenderers"},{"attributes":{},"id":"26627","type":"WheelZoomTool"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"26633","type":"PolyAnnotation"},{"attributes":{"source":{"id":"26723"}},"id":"26727","type":"CDSView"},{"attributes":{"line_dash":[6],"location":3.480769230769231},"id":"26728","type":"Span"},{"attributes":{},"id":"26739","type":"Selection"},{"attributes":{"toolbars":[{"id":"26634"},{"id":"26668"}],"tools":[{"id":"26624"},{"id":"26625"},{"id":"26626"},{"id":"26627"},{"id":"26628"},{"id":"26629"},{"id":"26630"},{"id":"26631"},{"id":"26658"},{"id":"26659"},{"id":"26660"},{"id":"26661"},{"id":"26662"},{"id":"26663"},{"id":"26664"},{"id":"26665"}]},"id":"26764","type":"ProxyToolbar"},{"attributes":{"toolbar":{"id":"26764"},"toolbar_location":"above"},"id":"26765","type":"ToolbarBox"},{"attributes":{"overlay":{"id":"26633"}},"id":"26628","type":"LassoSelectTool"},{"attributes":{"data_source":{"id":"26711"},"glyph":{"id":"26712"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26713"},"selection_glyph":null,"view":{"id":"26715"}},"id":"26714","type":"GlyphRenderer"},{"attributes":{"axis":{"id":"26616"},"ticker":null},"id":"26619","type":"Grid"},{"attributes":{},"id":"26749","type":"BasicTickFormatter"},{"attributes":{"bottom":{"value":1},"fill_color":{"value":"#ff7f0e"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26712","type":"VBar"},{"attributes":{},"id":"26629","type":"UndoTool"},{"attributes":{"text":"tau"},"id":"26703","type":"Title"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"26667","type":"PolyAnnotation"},{"attributes":{},"id":"26612","type":"LinearScale"},{"attributes":{"data":{"top":{"__ndarray__":"P/ADP/AD9z+SG7mRG7n2Py/0Qi/0QvU/eqEXeqEX9D9IbuRGbuT3P4If+IEf+PQ/MPRCL/RC9T+ZmZmZmZn3Pyd2Yid2YvQ/9kIv9EIv+D+4kRu5kRv7P7ATO7ETO/o/oBd6oRd6+D+mF3qhF3r4P1ZqpVZqpfk/9EIv9EIv+D9GbuRGbuT3P07sxE7sxPg/wA/8wA/8+z8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26756"},"selection_policy":{"id":"26757"}},"id":"26711","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"26705"},"glyph":{"id":"26706"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26707"},"selection_glyph":null,"view":{"id":"26709"}},"id":"26708","type":"GlyphRenderer"},{"attributes":{},"id":"26755","type":"UnionRenderers"},{"attributes":{"source":{"id":"26705"}},"id":"26709","type":"CDSView"},{"attributes":{"bottom":{"value":1},"fill_alpha":{"value":0.1},"fill_color":{"value":"#ff7f0e"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26713","type":"VBar"},{"attributes":{"data_source":{"id":"26723"},"glyph":{"id":"26724"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26725"},"selection_glyph":null,"view":{"id":"26727"}},"id":"26726","type":"GlyphRenderer"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"26666","type":"BoxAnnotation"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26707","type":"VBar"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"26632","type":"BoxAnnotation"},{"attributes":{"data":{"top":{"__ndarray__":"MzMzMzOzDUC4HoXrUTgPQDCW/GLJrwxA0GkDnTbQDEBtoNMGOu0LQOi0gU4baApAzszMzMzMC0DrUbgehWsLQClcj8L1qApAqA102kAnCkBH4XoUrkcKQMaSXyz5xQlA6LSBThtoCkAqXI/C9agKQClcj8L1qApACtejcD0KC0AGOm2g0wYKQMkvlvxiyQpAaQOdNtDpCkA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26745"},"selection_policy":{"id":"26746"}},"id":"26695","type":"ColumnDataSource"},{"attributes":{"below":[{"id":"26650"}],"center":[{"id":"26653"},{"id":"26657"},{"id":"26710"},{"id":"26716"},{"id":"26722"},{"id":"26728"}],"left":[{"id":"26654"}],"output_backend":"webgl","plot_height":331,"plot_width":441,"renderers":[{"id":"26708"},{"id":"26714"},{"id":"26720"},{"id":"26726"}],"title":{"id":"26731"},"toolbar":{"id":"26668"},"toolbar_location":null,"x_range":{"id":"26608"},"x_scale":{"id":"26646"},"y_range":{"id":"26610"},"y_scale":{"id":"26648"}},"id":"26643","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"26760","type":"Selection"},{"attributes":{},"id":"26746","type":"UnionRenderers"}],"root_ids":["26766"]},"title":"Bokeh Application","version":"2.2.3"}}';
                  var render_items = [{"docid":"164fe0d4-53e5-4a94-a7d0-307f478de958","root_ids":["26766"],"roots":{"26766":"bad0ed37-dc8c-4863-ad34-691786e3952a"}}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    var attempts = 0;
                    var timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
        function(Bokeh) {
        
        
        }
      ];
    
      function run_inline_js() {
        
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
        
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();